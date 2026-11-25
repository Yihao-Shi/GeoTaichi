"""Bounded-memory NPZ I/O for the prescribed explicit screw-pile case."""

from contextlib import contextmanager
import os
from zipfile import ZipFile

import numpy as np

CHUNK = 1_000_000
STATE_WIDTH = 14


@contextmanager
def atomic_npz(path):
    partial = path.with_suffix(".partial")
    try:
        with partial.open("wb") as stream:
            with ZipFile(stream, "w") as archive:
                yield archive
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        partial.unlink(missing_ok=True)


def write_array(archive, name, array):
    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, array, allow_pickle=False)


def save_frame(path, count, origin, scale, read_positions):
    with atomic_npz(path) as archive:
        write_array(archive, "origin", origin)
        write_array(archive, "scale", scale)
        for start in range(0, count, CHUNK):
            block = np.empty((min(CHUNK, count - start), 3), dtype=np.uint16)
            read_positions(start, scale, block)
            write_array(archive, f"position_{start}", block)


def frame_paths(directory):
    frames = {int(path.stem): path for path in directory.glob("*.npy") if path.stem.isdecimal()}
    frames.update({int(path.stem): path for path in directory.glob("*.npz") if path.stem.isdecimal()})
    return [frames[step] for step in sorted(frames)]


def load_frame(path, count):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    points = np.empty((count, 3), dtype=np.float32)
    with np.load(path, allow_pickle=False) as archive:
        origin = archive["origin"].astype(np.float32)
        scale = archive["scale"].astype(np.float32)
        for start in range(0, count, CHUNK):
            block = archive[f"position_{start}"]
            if block.dtype != np.uint16 or block.shape != points[start : start + CHUNK].shape:
                raise ValueError(f"Invalid position block {start} in {path}")
            dest = points[start : start + CHUNK]
            np.multiply(block, scale, out=dest)
            dest += origin
    return points
