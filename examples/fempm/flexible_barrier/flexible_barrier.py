"""Elastic FEM barrier and DP MPM soil, with implicit IPC contact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.mpm.Contact.FlexibleBarrier.coupled_barrier import main

if __name__ == "__main__":
    main("fempm")
