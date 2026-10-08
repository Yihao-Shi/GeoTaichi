import taichi as ti
import numpy as np
from taichi.lang.impl import current_cfg

from src.linear_solver.BlockSparsityPattern import BlockSparsityPatternCache
from src.utils.FieldIO import field_to_numpy_prefix, runtime_float_numpy_dtype


@ti.data_oriented
class HashReduction:
    def __init__(
        self,
        max_pairs_num,
        dim=3,
        max_nnz=0,
        hessian_size=6,
        pattern_cache=True,
        pattern_cache_extra_fraction=0.01,
        pattern_cache_max_age=25,
        device_reduction=None,
    ):
        max_pairs_num = int(max_pairs_num)
        max_nnz = int(max_nnz)
        if max_nnz == 0:
            max_nnz = max_pairs_num
        int32_max = int(np.iinfo(np.int32).max)
        if max_pairs_num <= 0 or max_nnz <= 0:
            raise ValueError("HashReduction raw-pair and reduced-nonzero capacities must " "be positive")
        if max_pairs_num > int32_max or max_nnz > int32_max:
            raise ValueError(
                "HashReduction capacities exceed Taichi's dense SNode/int32 "
                f"limit: raw_pairs={max_pairs_num}, reduced_nonzeros={max_nnz}, "
                f"limit={int32_max}"
            )
        self.dim = dim
        self.hessian_size = int(hessian_size)
        self.max_pairs_num = max_pairs_num
        runtime_arch = current_cfg().arch
        if device_reduction is None:
            # Runtime assembly is device-resident on every Taichi backend.
            # ``False`` remains an explicit test/reference oracle on CPU and
            # Metal, but is never selected by a production assembler.
            device_reduction = True
        elif runtime_arch == ti.cuda and not bool(device_reduction):
            raise RuntimeError(
                "device_reduction=False is not supported on CUDA: sparse "
                "block reduction must remain in Taichi device fields"
            )
        self.device_reduction = bool(device_reduction)
        self.numpy_float_dtype = runtime_float_numpy_dtype()
        hash_capacity = 1
        requested_hash_capacity = max(2, 2 * max_nnz) if self.device_reduction else 1
        while hash_capacity < requested_hash_capacity:
            hash_capacity <<= 1
        if hash_capacity > int32_max:
            raise ValueError(
                "HashReduction device hash-table capacity exceeds Taichi's "
                "dense SNode int32 limit: "
                f"reduced_nonzeros={max_nnz}, required_hash_slots="
                f"{hash_capacity}, limit={int32_max}; reduce max_nonzeros or "
                "use a tighter topology bound"
            )

        self.element_pair_num = ti.field(int, shape=1)
        self.blockI = ti.field(dtype=ti.i32, shape=max_pairs_num)
        self.blockJ = ti.field(dtype=ti.i32, shape=max_pairs_num)
        self.blockH = ti.Vector.field(self.hessian_size, dtype=float, shape=max_pairs_num)
        # Cached raw-triplet -> reduced-block mapping. Stable FEM/MPM stencils
        # validate this slot in O(1) and avoid open-addressing probes entirely.
        # Dynamic IPC patterns only hash the raw entries whose coordinate no
        # longer matches its cached reduced slot.
        raw_mapping_capacity = max_pairs_num if self.device_reduction else 1
        self.raw_output = ti.field(dtype=ti.i32, shape=raw_mapping_capacity)
        self.tripletI = ti.field(dtype=ti.i32, shape=max_nnz)
        self.tripletJ = ti.field(dtype=ti.i32, shape=max_nnz)
        self.tripletH = ti.Vector.field(self.hessian_size, dtype=float, shape=max_nnz)
        self.pattern_cache = BlockSparsityPatternCache(
            max_nnz,
            enabled=pattern_cache,
            extra_fraction=pattern_cache_extra_fraction,
            max_age=pattern_cache_max_age,
        )

        # Device-resident open-addressing table used by every Taichi backend. It
        # stores only coordinates and a block-output slot, while all changing
        # matrix values remain in ``tripletH``.  A load factor <= 0.5 keeps
        # probing short for collision-heavy contact patterns.
        self.device_hash_capacity = hash_capacity
        self.device_hash_mask = hash_capacity - 1
        self.device_hash_output = ti.field(dtype=ti.i32, shape=hash_capacity)
        self.device_miss_raw = ti.field(dtype=ti.i32, shape=max_pairs_num)
        self.device_last_seen = ti.field(dtype=ti.i32, shape=max_nnz)
        self.device_current_unique = ti.field(dtype=ti.i32, shape=1)
        self.device_new_entries = ti.field(dtype=ti.i32, shape=1)
        self.device_mapping_misses = ti.field(dtype=ti.i32, shape=1)
        self.device_overflow = ti.field(dtype=ti.i32, shape=1)
        self.fixed_count = 0
        self.fixed_rebuild_pending = False
        self.device_epoch = 0
        self.device_pattern_version = 0
        self.device_pattern_rebuilds = 0
        self.device_pattern_hits = 0
        self.device_pattern_initialized = False
        self.device_pattern_enabled = bool(pattern_cache)
        self.device_pattern_extra_fraction = float(pattern_cache_extra_fraction)
        self.device_pattern_max_age = int(pattern_cache_max_age)

        # Production uses Taichi fields for pattern lookup, reduction, and
        # Krylov matvec on every architecture.  The host path is an explicit
        # numerical oracle only.
        self.go = self.go_with_device_reduction if self.device_reduction else self.go_with_host_reduction

    def install_fixed_pattern(self, coordinates):
        """Reserve permanent reduced slots before the first assembly."""
        coordinates = np.asarray(coordinates)
        if not self.device_reduction or self.device_epoch or self.fixed_count:
            raise RuntimeError("fixed slots require a fresh device reducer")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.issubdtype(coordinates.dtype, np.integer):
            raise ValueError("fixed block coordinates must be an integer (n, 2) array")
        if np.any(coordinates < 0) or np.any(coordinates > np.iinfo(np.int32).max):
            raise ValueError("fixed block coordinates are outside int32 range")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("fixed block coordinates must be unique")
        if len(coordinates) > self.tripletI.shape[0]:
            raise ValueError("fixed block pattern exceeds reduced capacity")
        self.fixed_count = len(coordinates)
        if self.fixed_count:
            self._load_fixed_coordinates(np.ascontiguousarray(coordinates, dtype=np.int32))
            self._clear_device_pattern()

    @ti.kernel
    def _load_fixed_coordinates(self, coordinates: ti.types.ndarray(dtype=ti.i32, ndim=2)):
        for i in range(self.fixed_count):
            self.tripletI[i] = coordinates[i, 0]
            self.tripletJ[i] = coordinates[i, 1]

    def reset_fixed_values(self):
        if self.fixed_rebuild_pending:
            self._clear_device_pattern()
            self.device_pattern_initialized = False
            self.fixed_rebuild_pending = False
        self._reset_fixed_values()

    @ti.kernel
    def _reset_fixed_values(self):
        for i in range(self.fixed_count):
            self.tripletH[i] = ti.Vector.zero(float, self.hessian_size)

    def go_with_cpu(self, pairs_num):
        self.go_with_host_reduction(pairs_num)

    def go_with_gpu(self, pairs_num):
        self.go_with_device_reduction(pairs_num)

    def go_with_device_reduction(self, pairs_num):
        """Reduce blocks entirely in Taichi fields on the active backend.

        Existing block coordinates are found in a persistent open-addressing
        table, so stable Newton stencils only clear and atomically accumulate
        block values.  No triplet coordinate/value array crosses PCIe.
        """
        pairs_num = int(pairs_num)
        if pairs_num < 0 or pairs_num > self.max_pairs_num:
            raise ValueError(f"pairs_num={pairs_num} is outside [0, {self.max_pairs_num}]")
        reuse_raw_mapping = self.device_pattern_initialized and self.device_pattern_enabled
        if not reuse_raw_mapping:
            self._clear_device_pattern()
            self.device_pattern_initialized = True

        self.device_epoch += 1
        previous_nnz = int(self.element_pair_num[0])
        self._begin_device_reduction(previous_nnz, self.device_epoch)
        if pairs_num > 0:
            self._device_map_pattern(pairs_num, int(reuse_raw_mapping), previous_nnz)
        forced_rebuild = False
        if int(self.device_overflow[0]) != 0:
            # The current set may still fit even when the cached superset does
            # not. Drop stale entries and retry before reporting a real
            # capacity error.
            self._clear_device_pattern()
            self._begin_device_reduction(self.fixed_count, self.device_epoch)
            if pairs_num > 0:
                self._device_map_pattern(pairs_num, 0, 0)
            forced_rebuild = True
            if int(self.device_overflow[0]) != 0:
                failed_nnz = int(self.element_pair_num[0])
                self._clear_device_pattern()
                self.device_pattern_initialized = False
                raise RuntimeError(
                    "HashReduction device block-pattern table overflow: "
                    f"raw_pairs={pairs_num}, attempted_unique={failed_nnz}, "
                    f"capacity={self.tripletI.shape[0]}, hash_capacity={self.device_hash_capacity}. "
                    "Increase max_nonzeros."
                )
        if pairs_num > 0:
            # Kernel boundaries provide the global device synchronization
            # needed before lock-free readers consume newly published slots.
            self._device_reduce_values(pairs_num, self.device_epoch)
        if int(self.device_overflow[0]) != 0:
            self._clear_device_pattern()
            self.device_pattern_initialized = False
            raise RuntimeError("HashReduction device pattern lookup failed after insertion")

        pattern_nnz = int(self.element_pair_num[0])
        current_nnz = int(self.device_current_unique[0])
        new_entries = int(self.device_new_entries[0])
        expired = 0
        if pattern_nnz > current_nnz and self.device_pattern_max_age >= 0:
            expired = int(
                self._device_has_expired(
                    pattern_nnz,
                    self.device_epoch,
                    self.device_pattern_max_age,
                )
            )
        extra_budget = 0
        if self.device_pattern_extra_fraction > 0.0:
            extra_budget = max(
                1,
                int(np.ceil(self.device_pattern_extra_fraction * max(current_nnz, 1))),
            )
        too_many_stale = pattern_nnz - current_nnz > extra_budget

        # Compact to the exact current pattern only when the bounded superset
        # policy requires eviction. Values are always exact; eviction is
        # coarser but never changes the represented matrix.
        if forced_rebuild:
            self.device_pattern_version += 1
            self.device_pattern_rebuilds += 1
        elif self.fixed_count and (too_many_stale or expired):
            # Fixed slots already contain body values and dynamic additions.
            # Rebuild before the next assembly, never accumulate them twice.
            self.fixed_rebuild_pending = True
        elif (too_many_stale or expired) and pattern_nnz > 0:
            self._clear_device_pattern()
            self._begin_device_reduction(self.fixed_count, self.device_epoch)
            if pairs_num > 0:
                self._device_map_pattern(pairs_num, 0, 0)
            if int(self.device_overflow[0]) != 0:
                failed_nnz = int(self.element_pair_num[0])
                self._clear_device_pattern()
                self.device_pattern_initialized = False
                raise RuntimeError(
                    "HashReduction device pattern overflow while compacting "
                    f"current entries: raw_pairs={pairs_num}, "
                    f"attempted_unique={failed_nnz}, "
                    f"capacity={self.tripletI.shape[0]}"
                )
            if pairs_num > 0:
                self._device_reduce_values(pairs_num, self.device_epoch)
            if int(self.device_overflow[0]) != 0:
                self._clear_device_pattern()
                self.device_pattern_initialized = False
                raise RuntimeError("HashReduction device pattern lookup failed after compaction")
            self.device_pattern_version += 1
            self.device_pattern_rebuilds += 1
        elif new_entries > 0:
            self.device_pattern_version += 1
            self.device_pattern_rebuilds += 1
        else:
            self.device_pattern_hits += 1

    def go_with_host_reduction(self, pairs_num):
        pairs_num = int(pairs_num)
        if pairs_num < 0 or pairs_num > self.max_pairs_num:
            raise ValueError(f"pairs_num={pairs_num} is outside [0, {self.max_pairs_num}]")
        if pairs_num == 0:
            self.element_pair_num[0] = 0
            return

        block_i = np.ascontiguousarray(
            field_to_numpy_prefix(self.blockI, pairs_num),
            dtype=np.int32,
        )
        block_j = np.ascontiguousarray(
            field_to_numpy_prefix(self.blockJ, pairs_num),
            dtype=np.int32,
        )
        block_h = np.ascontiguousarray(
            field_to_numpy_prefix(self.blockH, pairs_num),
            dtype=np.float64,
        )
        # Fixed FEM/MPM stencils reserve one raw slot for every possible
        # local node pair. Diagonal and unused slots carry a negative block
        # coordinate, so they must never enter the sparsity pattern.
        valid = (block_i >= 0) & (block_j >= 0)
        block_i = np.ascontiguousarray(block_i[valid], dtype=np.int32)
        block_j = np.ascontiguousarray(block_j[valid], dtype=np.int32)
        block_h = np.ascontiguousarray(block_h[valid], dtype=np.float64)
        triplet_i, triplet_j, triplet_h = self.pattern_cache.reduce(
            block_i,
            block_j,
            block_h,
        )
        nnz = int(triplet_i.shape[0])
        if nnz > self.tripletI.shape[0]:
            raise RuntimeError(f"HashReduction reduced nonzeros {nnz} exceed max_nnz={self.tripletI.shape[0]}")
        self._upload_reduced_triplets(
            np.ascontiguousarray(triplet_i, dtype=np.int32),
            np.ascontiguousarray(triplet_j, dtype=np.int32),
            np.ascontiguousarray(triplet_h, dtype=self.numpy_float_dtype),
            nnz,
        )
        self.element_pair_num[0] = nnz

    def configure_pattern_cache(self, *, enabled=None, extra_fraction=None, max_age=None):
        self.pattern_cache.configure(
            enabled=enabled,
            extra_fraction=extra_fraction,
            max_age=max_age,
        )
        if enabled is not None:
            self.device_pattern_enabled = bool(enabled)
        if extra_fraction is not None:
            extra_fraction = float(extra_fraction)
            if extra_fraction < 0.0:
                raise ValueError("extra_fraction must be non-negative")
            self.device_pattern_extra_fraction = extra_fraction
        if max_age is not None:
            max_age = int(max_age)
            if max_age < 0:
                raise ValueError("max_age must be non-negative")
            self.device_pattern_max_age = max_age

    def pattern_cache_statistics(self):
        if not self.device_reduction:
            return self.pattern_cache.statistics()
        return {
            "enabled": self.device_pattern_enabled,
            "backend": "taichi_device",
            "pattern_nonzeros": int(self.element_pair_num[0]),
            "pattern_version": int(self.device_pattern_version),
            "pattern_rebuilds": int(self.device_pattern_rebuilds),
            "pattern_hits": int(self.device_pattern_hits),
            "reductions": int(self.device_epoch),
            "current_nonzeros": int(self.device_current_unique[0]),
            "last_new_entries": int(self.device_new_entries[0]),
            "last_mapping_misses": int(self.device_mapping_misses[0]),
        }

    @ti.kernel
    def _clear_device_pattern(self):
        self.element_pair_num[0] = self.fixed_count
        self.device_current_unique[0] = 0
        self.device_new_entries[0] = 0
        self.device_mapping_misses[0] = 0
        self.device_overflow[0] = 0
        for slot in self.device_hash_output:
            self.device_hash_output[slot] = -1
        for index in range(self.fixed_count, self.tripletI.shape[0]):
            self.tripletI[index] = 0
            self.tripletJ[index] = 0
            self.tripletH[index] = ti.Vector.zero(float, self.hessian_size)
            self.device_last_seen[index] = -1

        ti.loop_config(serialize=True)
        for index in range(self.fixed_count):
            slot = self._device_hash(self.tripletI[index], self.tripletJ[index])
            while self.device_hash_output[slot] >= 0:
                slot = (slot + 1) & self.device_hash_mask
            self.device_hash_output[slot] = index

    @ti.kernel
    def _begin_device_reduction(self, pattern_nnz: int, epoch: int):
        self.device_current_unique[0] = self.fixed_count
        self.device_new_entries[0] = 0
        self.device_mapping_misses[0] = 0
        self.device_overflow[0] = 0
        for index in range(self.fixed_count):
            self.device_last_seen[index] = epoch
        for index in range(self.fixed_count, pattern_nnz):
            self.tripletH[index] = ti.Vector.zero(float, self.hessian_size)

    @ti.func
    def _device_hash(self, block_i, block_j):
        row = ti.cast(block_i, ti.u32)
        col = ti.cast(block_j, ti.u32)
        mixed = (row * ti.u32(0x9E3779B1)) ^ (col * ti.u32(0x85EBCA77))
        mixed ^= mixed >> 16
        return ti.cast(mixed & ti.u32(self.device_hash_mask), ti.i32)

    def _device_map_pattern(self, pairs_num: int, reuse_raw_mapping: int, cached_pattern_nnz: int):
        self._device_collect_mapping_misses(pairs_num, reuse_raw_mapping, cached_pattern_nnz)
        misses = int(self.device_mapping_misses[0])
        if misses > 0:
            self._device_insert_mapping_misses(misses)

    @ti.kernel
    def _device_collect_mapping_misses(self, pairs_num: int, reuse_raw_mapping: int, cached_pattern_nnz: int):
        for raw in range(pairs_num):
            block_i = self.blockI[raw]
            block_j = self.blockJ[raw]
            if block_i < 0 or block_j < 0:
                # Reserved diagonal/unused stencil slot. Keep its raw mapping
                # invalid without treating it as a cache miss or an error.
                self.raw_output[raw] = -1
            else:
                output = -1
                if reuse_raw_mapping != 0:
                    mapped = self.raw_output[raw]
                    if (
                        0 <= mapped
                        and mapped < cached_pattern_nnz
                        and self.tripletI[mapped] == block_i
                        and self.tripletJ[mapped] == block_j
                    ):
                        output = mapped

                if output < 0:
                    miss = ti.atomic_add(self.device_mapping_misses[0], 1)
                    self.device_miss_raw[miss] = raw

                if output >= 0:
                    self.raw_output[raw] = output
                else:
                    self.raw_output[raw] = -1

    @ti.kernel
    def _device_insert_mapping_misses(self, miss_count: int):
        # ponytail: first-time pattern insertion is serial; replace with a
        # proven lock-free map only if pattern rebuilds become a measured cost.
        ti.loop_config(serialize=True)
        for miss in range(miss_count):
            raw = self.device_miss_raw[miss]
            block_i = self.blockI[raw]
            block_j = self.blockJ[raw]
            output = -1
            slot = self._device_hash(block_i, block_j)
            attempts = 0
            while output == -1 and attempts < self.device_hash_capacity:
                stored_output = self.device_hash_output[slot]
                if stored_output == -1:
                    output = self.element_pair_num[0]
                    if output < self.tripletI.shape[0]:
                        self.element_pair_num[0] += 1
                        self.tripletI[output] = block_i
                        self.tripletJ[output] = block_j
                        self.tripletH[output] = ti.Vector.zero(float, self.hessian_size)
                        self.device_last_seen[output] = -1
                        self.device_hash_output[slot] = output
                        self.device_new_entries[0] += 1
                    else:
                        self.device_overflow[0] = 1
                        output = -2
                elif stored_output < self.element_pair_num[0]:
                    if self.tripletI[stored_output] == block_i and self.tripletJ[stored_output] == block_j:
                        output = stored_output
                    else:
                        slot = (slot + 1) & self.device_hash_mask
                        attempts += 1
                else:
                    self.device_overflow[0] = 1
                    output = -2

            if attempts >= self.device_hash_capacity:
                self.device_overflow[0] = 1

            if output >= 0:
                self.raw_output[raw] = output
            else:
                self.raw_output[raw] = -1

    @ti.kernel
    def _device_reduce_values(self, pairs_num: int, epoch: int):
        for raw in range(pairs_num):
            if self.blockI[raw] >= 0 and self.blockJ[raw] >= 0:
                output = self.raw_output[raw]
                if 0 <= output and output < self.element_pair_num[0]:
                    old_epoch = ti.atomic_max(self.device_last_seen[output], epoch)
                    if old_epoch != epoch:
                        ti.atomic_add(self.device_current_unique[0], 1)
                    for component in range(self.hessian_size):
                        ti.atomic_add(
                            self.tripletH[output][component],
                            self.blockH[raw][component],
                        )
                else:
                    self.device_overflow[0] = 1

    @ti.kernel
    def _device_has_expired(self, pattern_nnz: int, epoch: int, max_age: int) -> int:
        expired = 0
        for index in range(pattern_nnz):
            if epoch - self.device_last_seen[index] > max_age:
                ti.atomic_max(expired, 1)
        return expired

    def set_triplets_from_numpy(self, block_i, block_j, block_h):
        block_i = np.asarray(block_i, dtype=np.int32)
        block_j = np.asarray(block_j, dtype=np.int32)
        block_h = np.asarray(block_h, dtype=self.numpy_float_dtype)
        assert block_i.ndim == 1 and block_j.ndim == 1
        assert block_i.shape == block_j.shape
        assert block_h.shape == (block_i.shape[0], self.hessian_size)
        pairs_num = int(block_i.shape[0])
        if pairs_num > self.max_pairs_num:
            raise ValueError(f"pairs_num={pairs_num} exceeds max_pairs_num={self.max_pairs_num}")
        self._upload_raw_triplets(
            np.ascontiguousarray(block_i),
            np.ascontiguousarray(block_j),
            np.ascontiguousarray(block_h),
            pairs_num,
        )
        return pairs_num

    def set_reduced_triplets_from_numpy(self, block_i, block_j, block_h):
        """Reduce host blocks and load them without full-capacity padding."""
        block_i = np.asarray(block_i, dtype=np.int32)
        block_j = np.asarray(block_j, dtype=np.int32)
        block_h = np.asarray(block_h, dtype=np.float64)
        valid = (block_i >= 0) & (block_j >= 0)
        block_i = block_i[valid]
        block_j = block_j[valid]
        block_h = block_h[valid]
        triplet_i, triplet_j, triplet_h = self.pattern_cache.reduce(block_i, block_j, block_h)
        nnz = int(triplet_i.size)
        if nnz > self.tripletI.shape[0]:
            raise RuntimeError(f"HashReduction reduced nonzeros {nnz} exceed max_nnz={self.tripletI.shape[0]}")
        self._upload_reduced_triplets(
            np.ascontiguousarray(triplet_i, dtype=np.int32),
            np.ascontiguousarray(triplet_j, dtype=np.int32),
            np.ascontiguousarray(triplet_h, dtype=self.numpy_float_dtype),
            nnz,
        )
        self.element_pair_num[0] = nnz
        if self.device_reduction:
            self.device_pattern_version += 1
            self.device_pattern_initialized = False
        return nnz

    def get_reduced_triplets_numpy(self):
        nnz = int(self.element_pair_num[0])
        return (
            field_to_numpy_prefix(self.tripletI, nnz).copy(),
            field_to_numpy_prefix(self.tripletJ, nnz).copy(),
            field_to_numpy_prefix(self.tripletH, nnz).copy(),
        )

    @ti.kernel
    def _upload_raw_triplets(
        self,
        block_i: ti.types.ndarray(),
        block_j: ti.types.ndarray(),
        block_h: ti.types.ndarray(),
        pairs_num: int,
    ):
        for index in range(pairs_num):
            self.blockI[index] = block_i[index]
            self.blockJ[index] = block_j[index]
            for component in range(self.hessian_size):
                self.blockH[index][component] = block_h[index, component]

    @ti.kernel
    def _upload_reduced_triplets(
        self,
        triplet_i: ti.types.ndarray(),
        triplet_j: ti.types.ndarray(),
        triplet_h: ti.types.ndarray(),
        nnz: int,
    ):
        for index in range(nnz):
            self.tripletI[index] = triplet_i[index]
            self.tripletJ[index] = triplet_j[index]
            for component in range(self.hessian_size):
                self.tripletH[index][component] = triplet_h[index, component]

    @staticmethod
    def reduce_triplets_reference(block_i, block_j, block_h):
        block_i = np.asarray(block_i, dtype=np.int32)
        block_j = np.asarray(block_j, dtype=np.int32)
        block_h = np.asarray(block_h, dtype=np.float64)
        valid = (block_i >= 0) & (block_j >= 0)
        block_i = block_i[valid]
        block_j = block_j[valid]
        block_h = block_h[valid]
        order = np.lexsort((block_j, block_i))
        block_i = block_i[order]
        block_j = block_j[order]
        block_h = block_h[order]

        out_i = []
        out_j = []
        out_h = []
        for idx in range(block_i.shape[0]):
            if idx == 0 or block_i[idx] != out_i[-1] or block_j[idx] != out_j[-1]:
                out_i.append(int(block_i[idx]))
                out_j.append(int(block_j[idx]))
                out_h.append(block_h[idx].copy())
            else:
                out_h[-1] += block_h[idx]
        reduced_h = np.asarray(out_h, dtype=np.float64) if out_h else np.zeros((0, block_h.shape[1]), dtype=np.float64)
        return (
            np.asarray(out_i, dtype=np.int32),
            np.asarray(out_j, dtype=np.int32),
            reduced_h,
        )

    @staticmethod
    def reduce_triplets_scipy(block_i, block_j, block_h, shape=None):
        from scipy.sparse import coo_matrix

        block_i = np.asarray(block_i, dtype=np.int32)
        block_j = np.asarray(block_j, dtype=np.int32)
        block_h = np.asarray(block_h, dtype=np.float64)
        valid = (block_i >= 0) & (block_j >= 0)
        block_i = block_i[valid]
        block_j = block_j[valid]
        block_h = block_h[valid]
        if shape is None:
            n = int(max(block_i.max(initial=0), block_j.max(initial=0)) + 1)
            shape = (n, n)
        pattern = coo_matrix((np.ones(block_i.shape[0]), (block_i, block_j)), shape=shape).tocsr().tocoo()
        hessian_size = block_h.shape[1]
        reduced_h = np.zeros((pattern.nnz, hessian_size), dtype=np.float64)
        for comp in range(hessian_size):
            reduced = coo_matrix((block_h[:, comp], (block_i, block_j)), shape=shape).tocsr()
            reduced_h[:, comp] = np.asarray(reduced[pattern.row, pattern.col]).reshape(-1)
        return pattern.row.astype(np.int32), pattern.col.astype(np.int32), reduced_h
