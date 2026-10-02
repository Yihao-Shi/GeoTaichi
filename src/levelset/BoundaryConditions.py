import taichi as ti

from src.utils.ScalarFunction import clamp


@ti.func
def get_periodic(index, grid_num):
    if index < 0:
        index += grid_num
    elif index > grid_num - 1:
        index -= grid_num
    return index

# =================================================================================== #
#                                 Linear exteploation                                 #
# =================================================================================== #
@ti.func
def get_index_linear(positionIndex, grid_num):
    index = clamp(1, grid_num - 2, int(positionIndex))
    indexp = clamp(0, grid_num - 1, index + 1)
    return index, indexp

@ti.func
def get_index_linear_periodic(positionIndex, grid_num):
    index = int(positionIndex)
    indexp = index + 1
    return get_periodic(index, grid_num), get_periodic(indexp, grid_num)

# =================================================================================== #
#                                 WENO3 exteploation                                  #
# =================================================================================== #
@ti.func
def get_index_weno3(positionIndex, grid_num):
    index = clamp(1, grid_num - 2, int(positionIndex))
    indexp = clamp(0, grid_num - 1, index + 1)
    indexm = clamp(0, grid_num - 1, index - 1)
    return indexm, index, indexp

@ti.func
def get_index_weno3_periodic(positionIndex, grid_num):
    index = int(positionIndex)
    indexp = index + 1
    indexm = index - 1
    return get_periodic(indexm, grid_num), get_periodic(index, grid_num), get_periodic(indexp, grid_num)

# =================================================================================== #
#                                 WENO4 exteploation                                  #
# =================================================================================== #
@ti.func
def get_index_weno4(positionIndex, grid_num):
    index = clamp(1, grid_num - 3, int(positionIndex))
    indexp = clamp(0, grid_num - 1, index + 1)
    indexpp = clamp(0, grid_num - 1, index + 2)
    indexm = clamp(0, grid_num - 1, index - 1)
    return indexm, index, indexp, indexpp

@ti.func
def get_index_weno4_periodic(positionIndex, grid_num):
    index = int(positionIndex)
    indexp = index + 1
    indexpp = index + 2
    indexm = index - 1
    return get_periodic(indexm, grid_num), get_periodic(index, grid_num), get_periodic(indexp, grid_num), get_periodic(indexpp, grid_num)

# =================================================================================== #
#                                 WENO5 exteploation                                  #
# =================================================================================== #
@ti.func
def get_index_weno5(positionIndex, grid_num):
    index = clamp(2, grid_num - 4, int(positionIndex))
    indexp = clamp(0, grid_num - 1, index + 1)
    indexpp = clamp(0, grid_num - 1, index + 2)
    indexmm = clamp(0, grid_num - 1, index - 2)
    indexm = clamp(0, grid_num - 1, index - 1)
    return indexmm, indexm, index, indexp, indexpp

@ti.func
def get_index_weno5_periodic(positionIndex, grid_num):
    index = int(positionIndex)
    indexp = index + 1
    indexpp = index + 2
    indexmm = index - 2
    indexm = index - 1
    return get_periodic(indexmm, grid_num), get_periodic(indexm, grid_num), get_periodic(index, grid_num), get_periodic(indexp, grid_num), get_periodic(indexpp, grid_num)