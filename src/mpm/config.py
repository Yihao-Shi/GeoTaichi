"""Import-sensitive settings shared by the Direct MPM and soft-particle kernels.

DIM specializes field shapes and kernel signatures. Set it before importing the
specialized engines; changing it does not resize existing fields or kernels.
DYNAMIC includes inertia in the implicit solve when True.
"""

DIM = 2
DYNAMIC = True


def set_dimension(dimension: int):
    global DIM
    if int(dimension) not in (2, 3):
        raise ValueError("MPM dimension must be 2 or 3")
    DIM = int(dimension)


def get_dimension():
    return int(DIM)
