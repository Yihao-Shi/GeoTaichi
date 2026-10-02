"""Pairwise IPC barrier and lagged-friction parameters for FEM--AffineBody."""

from src.fempm.contact.IPC import IPCModel


class AffineIPCModel(IPCModel):
    """IPC law indexed by ``(affine_body_id, fem_body_id)``."""

    model_name = "IPC"
    coupling_name = "FEM--AffineBody"
    first_body_name = "AffineBody"


__all__ = ["AffineIPCModel"]
