DIM = 2
DYNAMIC = True


def set_dimension(dimension: int):
    global DIM
    if int(dimension) not in (2, 3):
        raise ValueError("IGA dimension must be 2 or 3")
    DIM = int(dimension)


def get_dimension():
    return int(DIM)
