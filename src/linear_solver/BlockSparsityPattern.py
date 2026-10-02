"""Reusable block-sparsity plans for nonlinear sparse assembly.

The nonlinear solvers in GeoTaichi emit many duplicate dense blocks.  Their
values change at every Newton iteration, while the block coordinates are
usually unchanged (body stencils) or change only slightly (IPC contact).  This
module separates those two concerns: it caches the sorted block pattern and
only reduces the new values into the cached slots.

The cache retains a small, age-limited superset of disappearing entries.
Retained entries receive exactly zero values, so the cache changes storage and
symbolic-analysis work, never the represented matrix.
"""

from __future__ import annotations

import math

import numpy as np


_UINT32_LIMIT = 1 << 32
_UINT32_MASK = np.uint64(_UINT32_LIMIT - 1)


def pack_block_keys(block_i, block_j):
    """Pack non-negative 32-bit block coordinates into sortable uint64 keys."""
    block_i = np.asarray(block_i)
    block_j = np.asarray(block_j)
    if block_i.ndim != 1 or block_j.ndim != 1 or block_i.shape != block_j.shape:
        raise ValueError("block_i and block_j must be one-dimensional arrays of equal length")
    if block_i.size:
        if np.any(block_i < 0) or np.any(block_j < 0):
            raise ValueError("block coordinates must be non-negative")
        if np.any(block_i >= _UINT32_LIMIT) or np.any(block_j >= _UINT32_LIMIT):
            raise ValueError("block coordinates must fit in unsigned 32-bit integers")
    row = block_i.astype(np.uint64, copy=False)
    col = block_j.astype(np.uint64, copy=False)
    return (row << np.uint64(32)) | col


def unpack_block_keys(keys):
    """Return int32 row and column arrays for packed block keys."""
    keys = np.asarray(keys, dtype=np.uint64)
    return (
        (keys >> np.uint64(32)).astype(np.int32),
        (keys & _UINT32_MASK).astype(np.int32),
    )


class BlockSparsityPatternCache:
    """Cache a sorted block pattern and reduce changing values into it.

    Parameters
    ----------
    capacity:
        Maximum number of unique block entries accepted by the destination.
    enabled:
        Disabling the cache preserves the vectorized reduction but rebuilds the
        exact pattern on every call.
    extra_fraction:
        Maximum number of recently disappeared entries retained relative to
        the current pattern.
    max_age:
        Maximum number of reductions for which a disappeared entry is retained.
    """

    def __init__(
        self,
        capacity,
        *,
        enabled=True,
        extra_fraction=0.01,
        max_age=25,
    ):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.enabled = bool(enabled)
        self.extra_fraction = float(extra_fraction)
        self.max_age = int(max_age)
        if self.extra_fraction < 0.0:
            raise ValueError("extra_fraction must be non-negative")
        if self.max_age < 0:
            raise ValueError("max_age must be non-negative")

        self.keys = np.empty(0, dtype=np.uint64)
        self.ages = np.empty(0, dtype=np.int32)
        self.pattern_version = 0
        self.pattern_rebuilds = 0
        self.exact_mapping_hits = 0
        self.pattern_hits = 0
        self.reductions = 0
        self.peak_current_nonzeros = 0
        self.last_result = "empty"

        self._last_raw_keys = None
        self._last_inverse = None
        self._last_pattern_version = -1

    def configure(self, *, enabled=None, extra_fraction=None, max_age=None):
        """Update cache policy without discarding a still-valid pattern."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if extra_fraction is not None:
            extra_fraction = float(extra_fraction)
            if extra_fraction < 0.0:
                raise ValueError("extra_fraction must be non-negative")
            self.extra_fraction = extra_fraction
        if max_age is not None:
            max_age = int(max_age)
            if max_age < 0:
                raise ValueError("max_age must be non-negative")
            self.max_age = max_age

    def clear(self):
        """Discard the symbolic pattern and invalidate dependent plans."""
        if self.keys.size:
            self.pattern_version += 1
        self.keys = np.empty(0, dtype=np.uint64)
        self.ages = np.empty(0, dtype=np.int32)
        self._last_raw_keys = None
        self._last_inverse = None
        self._last_pattern_version = -1
        self.last_result = "cleared"

    def reduce(self, block_i, block_j, block_h):
        """Reduce duplicate blocks and return the cached pattern plus values."""
        block_i = np.asarray(block_i)
        block_j = np.asarray(block_j)
        block_h = np.asarray(block_h, dtype=np.float64)
        if block_h.ndim != 2:
            raise ValueError("block_h must have shape (pair_count, block_value_count)")
        if block_i.shape != block_j.shape or block_i.shape[0] != block_h.shape[0]:
            raise ValueError("block coordinates and values must have matching lengths")

        raw_keys = np.ascontiguousarray(pack_block_keys(block_i, block_j))
        self.reductions += 1

        if raw_keys.size == 0:
            return self._reduce_empty(block_h.shape[1])

        # Static FEM/MPM stencils take this path after the first Newton
        # iteration.  It avoids sorting and rebuilding any sparse structure.
        if (
            self.enabled
            and self._last_raw_keys is not None
            and self._last_pattern_version == self.pattern_version
            and not np.any(self.ages)
            and np.array_equal(raw_keys, self._last_raw_keys)
        ):
            inverse = self._last_inverse
            self.exact_mapping_hits += 1
            self.pattern_hits += 1
            self.last_result = "exact_mapping_hit"
            reduced_h = self._sum_values(inverse, block_h, self.keys.size)
            out_i, out_j = unpack_block_keys(self.keys)
            return out_i, out_j, reduced_h

        current_keys = np.unique(raw_keys)
        current_nnz = int(current_keys.size)
        self.peak_current_nonzeros = max(self.peak_current_nonzeros, current_nnz)
        if current_nnz > self.capacity:
            raise RuntimeError(f"reduced block nonzeros {current_nnz} exceed capacity={self.capacity}")

        target_keys, target_ages = self._target_pattern(current_keys)
        same_pattern = np.array_equal(target_keys, self.keys)
        if same_pattern:
            self.pattern_hits += 1
            self.last_result = "pattern_hit"
        else:
            self.keys = target_keys
            self.pattern_version += 1
            self.pattern_rebuilds += 1
            self.last_result = "pattern_rebuild"
        self.ages = target_ages

        inverse = np.searchsorted(self.keys, raw_keys)
        if np.any(inverse >= self.keys.size) or np.any(self.keys[inverse] != raw_keys):
            raise RuntimeError("internal block-pattern mapping failure")
        inverse = np.ascontiguousarray(inverse, dtype=np.int64)
        self._last_raw_keys = raw_keys.copy()
        self._last_inverse = inverse
        self._last_pattern_version = self.pattern_version

        reduced_h = self._sum_values(inverse, block_h, self.keys.size)
        out_i, out_j = unpack_block_keys(self.keys)
        return out_i, out_j, reduced_h

    def _reduce_empty(self, hessian_size):
        if not self.enabled or self.max_age == 0 or self.extra_fraction == 0.0:
            target_keys = np.empty(0, dtype=np.uint64)
            target_ages = np.empty(0, dtype=np.int32)
        else:
            aged = self.ages.astype(np.int64) + 1
            eligible = aged <= self.max_age
            extra_budget = min(
                self.capacity,
                max(1, int(math.ceil(self.extra_fraction * max(self.peak_current_nonzeros, 1)))),
            )
            candidate_keys = self.keys[eligible]
            candidate_ages = aged[eligible]
            if candidate_keys.size > extra_budget:
                order = np.lexsort((candidate_keys, candidate_ages))[:extra_budget]
                candidate_keys = candidate_keys[order]
                candidate_ages = candidate_ages[order]
            sort_order = np.argsort(candidate_keys)
            target_keys = candidate_keys[sort_order].copy()
            target_ages = candidate_ages[sort_order].astype(np.int32, copy=False)

        if not np.array_equal(target_keys, self.keys):
            self.keys = target_keys
            self.pattern_version += 1
            self.pattern_rebuilds += 1
            self.last_result = "pattern_rebuild"
        else:
            self.pattern_hits += 1
            self.last_result = "pattern_hit"
        self.ages = target_ages
        self._last_raw_keys = np.empty(0, dtype=np.uint64)
        self._last_inverse = np.empty(0, dtype=np.int64)
        self._last_pattern_version = self.pattern_version
        out_i, out_j = unpack_block_keys(self.keys)
        return out_i, out_j, np.zeros((self.keys.size, hessian_size), dtype=np.float64)

    def _target_pattern(self, current_keys):
        if not self.enabled:
            return current_keys.copy(), np.zeros(current_keys.size, dtype=np.int32)

        stale_keys = np.empty(0, dtype=np.uint64)
        stale_ages = np.empty(0, dtype=np.int64)
        if self.keys.size and self.max_age > 0 and self.extra_fraction > 0.0:
            positions = np.searchsorted(current_keys, self.keys)
            present = positions < current_keys.size
            present[present] &= current_keys[positions[present]] == self.keys[present]
            aged = self.ages.astype(np.int64) + 1
            eligible = (~present) & (aged <= self.max_age)
            stale_keys = self.keys[eligible]
            stale_ages = aged[eligible]

            extra_budget = max(1, int(math.ceil(self.extra_fraction * max(current_keys.size, 1))))
            extra_budget = min(extra_budget, self.capacity - current_keys.size)
            if stale_keys.size > extra_budget:
                # Keep the youngest entries first; keys make the choice
                # deterministic when ages are equal.
                order = np.lexsort((stale_keys, stale_ages))[:extra_budget]
                stale_keys = stale_keys[order]
                stale_ages = stale_ages[order]

        target_keys = np.sort(np.concatenate((current_keys, stale_keys)))
        if target_keys.size > self.capacity:
            raise RuntimeError(f"cached block nonzeros {target_keys.size} exceed capacity={self.capacity}")
        target_ages = np.zeros(target_keys.size, dtype=np.int32)
        if stale_keys.size:
            target_ages[np.searchsorted(target_keys, stale_keys)] = stale_ages.astype(np.int32)
        return target_keys, target_ages

    @staticmethod
    def _sum_values(inverse, block_h, output_size):
        reduced_h = np.empty((int(output_size), block_h.shape[1]), dtype=np.float64)
        for component in range(block_h.shape[1]):
            reduced_h[:, component] = np.bincount(
                inverse,
                weights=block_h[:, component],
                minlength=int(output_size),
            )
        return reduced_h

    def statistics(self):
        """Return stable counters suitable for solver profiling and tests."""
        return {
            "enabled": self.enabled,
            "pattern_nonzeros": int(self.keys.size),
            "pattern_version": int(self.pattern_version),
            "pattern_rebuilds": int(self.pattern_rebuilds),
            "pattern_hits": int(self.pattern_hits),
            "exact_mapping_hits": int(self.exact_mapping_hits),
            "reductions": int(self.reductions),
            "peak_current_nonzeros": int(self.peak_current_nonzeros),
            "retained_nonzeros": int(np.count_nonzero(self.ages)),
            "last_result": self.last_result,
        }
