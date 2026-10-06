"""Audit sparse native output produced by this example family."""

from __future__ import annotations

from pathlib import Path


def collect_native_output(
    root: Path,
    expected_frames: int,
    required_families: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Count native artifacts and report completeness for present body types."""

    root = Path(root)
    patterns = {
        "fem_vtu_count": "vtks/FEM*.vtu",
        "lsdem_surface_vtu_count": "vtks/GraphicLSDEMSurface*.vtu",
        "lsdem_body_npz_count": "particles/LSDEMRigid*.npz",
        "lsdem_surface_npz_count": "particles/LSDEMSurface*.npz",
        "coupled_contact_npz_count": "FEDEMcontacts/FEDEMContact*.npz",
        "checkpoint_npz_count": "checkpoints/FEDEMCheckpoint*.npz",
    }
    evidence: dict[str, object] = {name: len(list(root.glob(pattern))) for name, pattern in patterns.items()}
    required = tuple(patterns) if required_families is None else required_families
    unknown = tuple(name for name in required if name not in patterns)
    if unknown:
        raise ValueError(f"unknown native-output families: {unknown}")
    evidence["expected_frame_count"] = int(expected_frames)
    evidence["required_families"] = list(required)
    evidence["complete"] = all(int(evidence[name]) >= int(expected_frames) for name in required)
    evidence["root"] = str(root)
    return evidence


__all__ = ["collect_native_output"]
