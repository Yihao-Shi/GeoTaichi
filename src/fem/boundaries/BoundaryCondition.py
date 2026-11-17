"""Dirichlet and Neumann boundary conditions for FEM meshes."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DirichletEntry:
    nodes: object
    components: object
    value: object


@dataclass
class LoadEntry:
    kind: str
    value: object
    nodes: object = None
    selector: object = None
    total: bool = False
    pressure: bool = False
