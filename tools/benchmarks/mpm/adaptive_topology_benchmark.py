"""Compare warmed device remeshing with the retained host topology reference.

The host timing excludes its field downloads/uploads, giving a conservative
comparison. Neither timing includes particle marking, splitting, or physics.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.mpm.elements.AdaptiveNodeMap import (
    AdaptiveNodeMap,
    build_bridging_node_weights,
    build_hanging_constraint_table,
)


def snapshots(cells, max_level, events):
    result = []
    for event in range(events):
        levels = np.zeros(tuple(cells), dtype=np.uint8)
        width = 2 + (cells[0] - 4) * (event + 1) // events
        region = [slice(1, width)] + [slice(1, max(3, size // 2)) for size in cells[1:]]
        levels[tuple(region)] = 1
        if max_level == 2 and width > 4:
            inner = [slice(2, width - 1)] + [slice(2, max(3, size // 2) - 1) for size in cells[1:]]
            levels[tuple(inner)] = 2
        result.append(levels.flatten(order="F"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--max-level", type=int, choices=[1, 2], default=2)
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arch", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--fp", choices=["f32", "f64"], default="f64")
    parser.add_argument("--mode", choices=["shape", "bridging"], default="shape")
    args = parser.parse_args()
    if len(args.cells) not in (2, 3) or min(args.cells) < 6 or args.events < 1 or args.repeats < 1:
        parser.error("choose 2D/3D cells >= 6, with positive events and repeats")
    if args.mode == "bridging" and args.max_level != 1:
        parser.error("bridging requires --max-level 1")
    ti.init(arch=getattr(ti, args.arch), default_fp=getattr(ti, args.fp), offline_cache=False)
    dimension, bridging = len(args.cells), args.mode == "bridging"
    cells = np.asarray(args.cells)
    levels = snapshots(cells, args.max_level, args.events)
    nodes = AdaptiveNodeMap(cells, 0, 1.0, bridging, args.max_level)
    initial_mapping = nodes.logical_to_compact.copy()
    initial_compact = nodes.compact_to_logical.copy()
    initial_count = nodes.next_compact_id
    mapping = ti.field(int, shape=nodes.full_logical_node_count)
    compact = ti.field(int, shape=nodes.capacity)
    fine_mapping = ti.field(int, shape=nodes.full_logical_node_count) if bridging else None
    nodes.bind_fields(mapping, compact, fine_mapping)
    refined = ti.field(ti.u8, shape=int(np.prod(cells)))
    lookup = ti.field(int, shape=nodes.capacity)
    count = ti.field(int, shape=())
    slaves = ti.field(int, shape=nodes.capacity)
    masters = ti.Vector.field(1 << dimension, int, shape=nodes.capacity)
    weights = ti.Vector.field(1 << dimension, float, shape=nodes.capacity)
    touched_count = ti.field(int, shape=())
    touched = ti.field(int, shape=nodes.capacity)
    bridge_weights = ti.field(float, shape=nodes.coarse_node_count)

    def device_event():
        nodes.ensure_refined_cells_device(refined, "GIMP")
        if bridging:
            nodes.device.build_bridge_weights(refined, 3, bridge_weights)
        else:
            nodes.device.build_constraints(
                refined, mapping, count, slaves, lookup, masters, weights, touched_count, touched
            )
        ti.sync()

    # Compile once on the same field identities that will be timed.
    refined.from_numpy(levels[-1])
    device_event()
    host_times, device_times = [], []
    reference = None
    for _ in range(args.repeats):
        reference = AdaptiveNodeMap(cells, 0, 1.0, bridging, args.max_level)
        elapsed = 0.0
        for snapshot in levels:
            start = time.perf_counter()
            reference.ensure_refined_cells(snapshot, "GIMP")
            if bridging:
                build_bridging_node_weights(snapshot, cells, 3)
            else:
                build_hanging_constraint_table(
                    snapshot,
                    cells,
                    reference.fine_gnum,
                    reference.logical_to_compact,
                    reference.compact_to_logical,
                    reference.next_compact_id,
                    max_level=args.max_level,
                )
            elapsed += time.perf_counter() - start
        host_times.append(elapsed)

        mapping.from_numpy(initial_mapping)
        compact.from_numpy(initial_compact)
        if bridging:
            fine_mapping.fill(-1)
        nodes.device.node_count[None] = initial_count
        nodes.device.cell_levels.fill(0)
        elapsed = 0.0
        for snapshot in levels:
            refined.from_numpy(snapshot)  # Mocks a completed device marker, outside the timed event.
            ti.sync()
            start = time.perf_counter()
            device_event()
            elapsed += time.perf_counter() - start
        device_times.append(elapsed)

    active = int(nodes.device.node_count[None])
    np.testing.assert_array_equal(
        compact.to_numpy()[:active], reference.compact_to_logical[: reference.next_compact_id]
    )
    host_seconds, device_seconds = statistics.median(host_times), statistics.median(device_times)
    print(
        json.dumps(
            {
                "backend": str(ti.lang.impl.current_cfg().arch),
                "precision": args.fp,
                "cells": args.cells,
                "max_level": args.max_level,
                "mode": args.mode,
                "events": args.events,
                "repeats": args.repeats,
                "active_nodes": active,
                "host_reference_seconds": host_seconds,
                "device_seconds": device_seconds,
                "speedup": host_seconds / device_seconds,
                "scope": "node allocation and interface topology; host transfers and JIT excluded",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
