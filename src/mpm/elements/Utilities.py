from src.utils.errors import CapacityError, SimulationStateError
from src.mpm.elements.ElementNodesTHBGenerate import element_list
from src.mpm.elements.ElementNodesTHBGenerate import Element
from src.mpm.elements.ElementNodesTHBGenerate import Node
from src.mpm.elements.ElementNodesTHBGenerate import particle_information
from src.mpm.elements.ElementNodesTHBGenerate import THB_MAX_INFLUENCE_NODES
from src.mpm.elements.ElementNodesTHBGenerate import level
from src.mpm.elements.ElementNodesTHBGenerate import element_num
from src.mpm.elements.ElementNodesTHBGenerate import node_num
from src.mpm.elements.ElementNodesTHBGenerate import particle_num
from src.mpm.elements.ElementNodesTHBGenerate import node_list
from src.mpm.elements.ElementNodesTHBGenerate import npelem
import numpy as np
import taichi as ti
from collections import deque
from itertools import product


def elem_nodes_gen(baseLength, modelRange, level, refineBound, write_debug_files=False):
    global element_list, node_list, element_num, node_num, particle_information

    Nx = int((modelRange[0][1] - modelRange[0][0]) / baseLength + 0.5) + 1
    Ny = int((modelRange[1][1] - modelRange[1][0]) / baseLength + 0.5) + 1

    base_node_count = Nx * Ny
    base_element_count = (Nx - 1) * (Ny - 1)
    # A refined quadrilateral replaces one leaf with four leaves.  Allocate a
    # conservative problem-sized upper bound instead of two million Python
    # objects for every THB case.  Each division creates at most five nodes;
    # shared edge nodes only reduce the actual count.
    max_element_count = base_element_count * 4**level
    max_division_count = base_element_count * sum(4**layer for layer in range(level))
    max_node_count = base_node_count + 5 * max_division_count
    node_list = [Node() for _ in range(max_node_count + 1)]
    element_list = [Element() for _ in range(max_element_count + 1)]

    for i in range(1, max_element_count + 1):
        element_list[i].childElem = [0] * level

    for iy in range(1, Ny + 1):
        for ix in range(1, Nx + 1):
            i = (iy - 1) * Nx + ix
            node_list[i].coord[0] = (ix - 1) * baseLength + modelRange[0][0]
            node_list[i].coord[1] = (iy - 1) * baseLength + modelRange[1][0]
            node_list[i].level = [0] * 2

    node_num = Nx * Ny

    for iy in range(1, Ny):
        for ix in range(1, Nx):
            i = (iy - 1) * (Nx - 1) + ix
            node1 = (iy - 1) * Nx + ix
            element_list[i].includeNode[0] = node1
            element_list[i].includeNode[1] = node1 + 1
            element_list[i].includeNode[2] = node1 + 1 + Nx
            element_list[i].includeNode[3] = node1 + Nx

            element_list[i].center[0] = 0.5 * (node_list[node1].coord[0] + node_list[node1 + 1].coord[0])
            element_list[i].center[1] = 0.5 * (node_list[node1].coord[1] + node_list[node1 + Nx].coord[1])

            if iy > 1:
                element_list[i].neighbor[0] = i + 1 - Nx
            if ix < Nx - 1:
                element_list[i].neighbor[1] = i + 1
            if iy < Ny - 1:
                element_list[i].neighbor[2] = i + Nx - 1
            if ix > 1:
                element_list[i].neighbor[3] = i - 1

            element_list[i].level = 0

    element_num = (Nx - 1) * (Ny - 1)

    print(f"Numbers of Element and Node: {element_num}, {node_num}")

    for l in range(1, level + 1):
        L = baseLength / (2**l)
        node_coord = [[L, 0.0], [2.0 * L, L], [L, 2.0 * L], [0.0, L]]

        for i in range(1, element_num + 1):
            if element_list[i].level != (l - 1):
                continue
            elem_center = element_list[i].center

            if (
                refineBound[l - 1][0][0] <= elem_center[0] <= refineBound[l - 1][0][1]
                and refineBound[l - 1][1][0] <= elem_center[1] <= refineBound[l - 1][1][1]
            ):
                element_list[i].divide = True

            if element_list[i].divide:
                nb_newnode = 0
                for j in range(4, 8):
                    if element_list[i].includeNode[j] < 1:
                        nb_newnode += 1
                        element_list[i].includeNode[j] = node_num + nb_newnode
                        node_list[node_num + nb_newnode].coord = [
                            node_list[element_list[i].includeNode[0]].coord[k] + node_coord[j - 4][k] for k in range(2)
                        ]

                nb_newnode += 1
                node_list[node_num + nb_newnode].coord = [
                    node_list[element_list[i].includeNode[0]].coord[k] + baseLength / (2**l) for k in range(2)
                ]

                element_list[i].center = [
                    node_list[element_list[i].includeNode[0]].coord[k] + 0.5 * baseLength / (2**l) for k in range(2)
                ]
                element_list[element_num + 1].center = [
                    node_list[element_list[i].includeNode[0]].coord[0] + 1.5 * baseLength / (2**l),
                    node_list[element_list[i].includeNode[0]].coord[1] + 0.5 * baseLength / (2**l),
                ]
                element_list[element_num + 2].center = [
                    node_list[element_list[i].includeNode[0]].coord[k] + 1.5 * baseLength / (2**l) for k in range(2)
                ]
                element_list[element_num + 3].center = [
                    node_list[element_list[i].includeNode[0]].coord[0] + 0.5 * baseLength / (2**l),
                    node_list[element_list[i].includeNode[0]].coord[1] + 1.5 * baseLength / (2**l),
                ]

                element_list[element_num + 1].includeNode[0] = element_list[i].includeNode[4]
                element_list[element_num + 1].includeNode[1] = element_list[i].includeNode[1]
                element_list[element_num + 1].includeNode[2] = element_list[i].includeNode[5]
                element_list[element_num + 1].includeNode[3] = node_num + nb_newnode

                element_list[element_num + 2].includeNode[0] = node_num + nb_newnode
                element_list[element_num + 2].includeNode[1] = element_list[i].includeNode[5]
                element_list[element_num + 2].includeNode[2] = element_list[i].includeNode[2]
                element_list[element_num + 2].includeNode[3] = element_list[i].includeNode[6]

                element_list[element_num + 3].includeNode[0] = element_list[i].includeNode[7]
                element_list[element_num + 3].includeNode[1] = node_num + nb_newnode
                element_list[element_num + 3].includeNode[2] = element_list[i].includeNode[6]
                element_list[element_num + 3].includeNode[3] = element_list[i].includeNode[3]

                elnode = [6, 7, 4, 5]
                for j in range(4):
                    e = element_list[i].neighbor[j]
                    if e == 0 or element_list[e].divide:
                        continue
                    element_list[e].includeNode[elnode[j]] = element_list[i].includeNode[j + 4]

                e = element_list[i].neighbor[0]
                if e != 0:
                    element_list[element_num + 1].neighbor[0] = e
                    if element_list[e].divide:
                        e1 = e
                        e2 = element_list[e1].childElem[l - 1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        element_list[i].neighbor[0] = e4
                        element_list[e4].neighbor[2] = i
                        element_list[element_num + 1].neighbor[0] = e3
                        element_list[e3].neighbor[2] = element_num + 1

                e = element_list[i].neighbor[1]
                if e != 0:
                    element_list[element_num + 1].neighbor[1] = e
                    element_list[element_num + 2].neighbor[1] = e
                    if element_list[e].divide:
                        e1 = e
                        e2 = element_list[e1].childElem[l - 1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        element_list[element_num + 1].neighbor[1] = e1
                        element_list[e1].neighbor[3] = element_num + 1
                        element_list[element_num + 2].neighbor[1] = e4
                        element_list[e4].neighbor[3] = element_num + 2

                e = element_list[i].neighbor[2]
                if e != 0:
                    element_list[element_num + 2].neighbor[2] = e
                    element_list[element_num + 3].neighbor[2] = e
                    if element_list[e].divide:
                        e1 = e
                        e2 = element_list[e1].childElem[l - 1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        element_list[element_num + 2].neighbor[2] = e2
                        element_list[e2].neighbor[0] = element_num + 2
                        element_list[element_num + 3].neighbor[2] = e1
                        element_list[e1].neighbor[0] = element_num + 3

                e = element_list[i].neighbor[3]
                if e != 0:
                    element_list[element_num + 3].neighbor[3] = e
                    if element_list[e].divide:
                        e1 = e
                        e2 = element_list[e1].childElem[l - 1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        element_list[i].neighbor[3] = e2
                        element_list[e2].neighbor[1] = i
                        element_list[element_num + 3].neighbor[3] = e3
                        element_list[e3].neighbor[1] = element_num + 3

                element_list[i].neighbor[1] = element_num + 1
                element_list[i].neighbor[2] = element_num + 3
                element_list[element_num + 1].neighbor[2] = element_num + 2
                element_list[element_num + 1].neighbor[3] = i
                element_list[element_num + 2].neighbor[0] = element_num + 1
                element_list[element_num + 2].neighbor[3] = element_num + 3
                element_list[element_num + 3].neighbor[0] = i
                element_list[element_num + 3].neighbor[1] = element_num + 2

                element_list[i].includeNode[1] = element_list[i].includeNode[4]
                element_list[i].includeNode[2] = node_num + nb_newnode
                element_list[i].includeNode[3] = element_list[i].includeNode[7]
                element_list[i].includeNode[4:8] = [0] * 4

                element_list[i].childElem[l - 1] = element_num + 1

                element_list[i].level = l
                for j in range(1, 4):
                    element_list[element_num + j].level = l

                node_num += nb_newnode
                element_num += 3

        for i in range(1, element_num + 1):
            element_list[i].divide = False

    for e in range(1, element_num + 1):
        elem = element_list[e]
        element_level = elem.level
        for i in range(4):
            inode = elem.includeNode[i]
            node_list[inode].level = [element_level] * 2
            node_list[inode].Ntype = [3] * 2

    for e in range(1, element_num + 1):
        elem = element_list[e]
        element_level = elem.level
        nei_elem = elem.neighbor

        e1 = nei_elem[0]
        l1 = element_list[e1].level if e1 != 0 else -1
        e2 = nei_elem[1]
        l2 = element_list[e2].level if e2 != 0 else -1
        e3 = nei_elem[2]
        l3 = element_list[e3].level if e3 != 0 else -1
        e4 = nei_elem[3]
        l4 = element_list[e4].level if e4 != 0 else -1

        if 0 in nei_elem:
            if nei_elem[3] == 0:
                if element_level < l3 and l3 > 0:
                    None
                elif element_level < l1 and l1 >= 0:
                    None
                else:
                    node_list[elem.includeNode[0]].Ntype[0] = 1
                    node_list[elem.includeNode[3]].Ntype[0] = 1
                    node_list[elem.includeNode[1]].Ntype[0] = 2
                    node_list[elem.includeNode[2]].Ntype[0] = 2
                if element_level > l1 >= 0:
                    for k in range(0, 2):
                        node_list[elem.includeNode[k]].Ntype[1] = 4
                    for k in range(2, 4):
                        node_list[elem.includeNode[k]].Ntype[1] = 5
                    for k in range(0, 4):
                        node_list[elem.includeNode[k]].level[1] = element_level - 1
                    node_list[elem.includeNode[0]].level[0] = element_level
                elif element_level > l3 >= 0:
                    for k in range(2, 4):
                        node_list[elem.includeNode[k]].Ntype[1] = 7
                    for k in range(0, 2):
                        node_list[elem.includeNode[k]].Ntype[1] = 6
                    for k in range(0, 4):
                        node_list[elem.includeNode[k]].level[1] = element_level - 1
                    node_list[element_list[e2].includeNode[2]].Ntype[0] = 3
            if nei_elem[1] == 0:
                for k in range(1, 3):
                    node_list[elem.includeNode[k]].Ntype[0] = 9
                node_list[elem.includeNode[0]].Ntype[0] = 8
                node_list[elem.includeNode[3]].Ntype[0] = 8
            if nei_elem[0] == 0:
                if element_level < l2 and l2 > 0:
                    None
                elif element_level < l4 and l4 >= 0:
                    None
                else:
                    node_list[elem.includeNode[0]].Ntype[1] = 1
                    node_list[elem.includeNode[1]].Ntype[1] = 1
                    node_list[elem.includeNode[2]].Ntype[1] = 2
                    node_list[elem.includeNode[3]].Ntype[1] = 2
                if element_level > l4 >= 0:
                    node_list[elem.includeNode[0]].Ntype[0] = 4
                    node_list[elem.includeNode[3]].Ntype[0] = 4
                    node_list[elem.includeNode[1]].Ntype[0] = 5
                    node_list[elem.includeNode[2]].Ntype[0] = 5
                    for k in range(0, 4):
                        node_list[elem.includeNode[k]].level[0] = element_level - 1
                    node_list[element_list[e3].includeNode[3]].Ntype[1] = 3
                elif element_level > l2 >= 0:
                    node_list[elem.includeNode[1]].Ntype[0] = 7
                    node_list[elem.includeNode[2]].Ntype[0] = 7
                    node_list[elem.includeNode[0]].Ntype[0] = 6
                    node_list[elem.includeNode[3]].Ntype[0] = 6
                    for k in range(0, 4):
                        node_list[elem.includeNode[k]].level[0] = element_level - 1
                    node_list[element_list[e3].includeNode[2]].Ntype[1] = 3
            if nei_elem[2] == 0:
                for k in range(2, 4):
                    node_list[elem.includeNode[k]].Ntype[1] = 9
                for k in range(0, 2):
                    node_list[elem.includeNode[k]].Ntype[1] = 8
        else:
            if element_level > l1:
                for k in range(0, 2):
                    node_list[elem.includeNode[k]].Ntype[1] = 4
                for k in range(2, 4):
                    node_list[elem.includeNode[k]].Ntype[1] = 5
                for k in range(0, 4):
                    node_list[elem.includeNode[k]].level[1] = element_level - 1
            elif element_level > l3:
                for k in range(2, 4):
                    node_list[elem.includeNode[k]].Ntype[1] = 7
                for k in range(0, 2):
                    node_list[elem.includeNode[k]].Ntype[1] = 6
                for k in range(0, 4):
                    node_list[elem.includeNode[k]].level[1] = element_level - 1
            if element_level > l4:
                node_list[elem.includeNode[0]].Ntype[0] = 4
                node_list[elem.includeNode[3]].Ntype[0] = 4
                for k in range(1, 3):
                    node_list[elem.includeNode[k]].Ntype[0] = 5
                for k in range(0, 4):
                    node_list[elem.includeNode[k]].level[0] = element_level - 1
            elif element_level > l2:
                node_list[elem.includeNode[0]].Ntype[0] = 6
                node_list[elem.includeNode[3]].Ntype[0] = 6
                for k in range(1, 3):
                    node_list[elem.includeNode[k]].Ntype[0] = 7
                for k in range(0, 4):
                    node_list[elem.includeNode[k]].level[0] = element_level - 1

    # >>>>>>>>>> Assign elements' influenced nodes <<<<<<<<<<
    node_infl_zone = [[0.0, 0.0], [0.0, 0.0]]
    threshold = baseLength * 1.0e-3
    for i in range(1, node_num + 1):
        x = node_list[i].coord
        for j in range(2):
            k = node_list[i].level[j]
            width = baseLength / (2**k)
            node_infl_zone[j][0] = x[j] - 2.0 * width
            node_infl_zone[j][1] = x[j] + 2.0 * width
            nodetype = node_list[i].Ntype

            if nodetype[j] == 4:
                node_infl_zone[j][0] = x[j] - 2.0 * width
                node_infl_zone[j][1] = x[j] + 1.5 * width
            elif nodetype[j] == 5:
                node_infl_zone[j][0] = x[j] - 1.5 * width
                node_infl_zone[j][1] = x[j] + 1.0 * width
            elif nodetype[j] == 6:
                node_infl_zone[j][0] = x[j] - 1.0 * width
                node_infl_zone[j][1] = x[j] + 1.5 * width
            elif nodetype[j] == 7:
                node_infl_zone[j][0] = x[j] - 1.5 * width
                node_infl_zone[j][1] = x[j] + 2.0 * width

        for e in range(1, element_num + 1):
            if (
                node_infl_zone[0][0] - threshold <= element_list[e].center[0] <= node_infl_zone[0][1] + threshold
                and node_infl_zone[1][0] - threshold <= element_list[e].center[1] <= node_infl_zone[1][1] + threshold
            ):
                element_list[e].nbInfNode += 1
                if element_list[e].nbInfNode > THB_MAX_INFLUENCE_NODES:
                    raise CapacityError(
                        "THB influenced-node capacity exceeded: "
                        f"element={e}, center={element_list[e].center}, "
                        f"level={element_list[e].level}, node={i}, "
                        f"count={element_list[e].nbInfNode}, "
                        f"capacity={THB_MAX_INFLUENCE_NODES}. "
                        "Increase the THB support capacity or adjust the "
                        "refinement patch so each particle support fits the "
                        "allocated THB kernel width."
                    )
                element_list[e].influenNode[element_list[e].nbInfNode - 1] = i

    # check node information
    xx = np.zeros((node_num, 2))
    xp = np.zeros((node_num, 2), dtype=int)
    xl = np.zeros((node_num, 2), dtype=int)
    for i in range(1, node_num + 1):
        xx[i - 1][:] = node_list[i].coord[:]
        xp[i - 1][:] = node_list[i].Ntype[:]
        xl[i - 1][:] = node_list[i].level[:]

    # plt.figure(1)
    # plt.scatter(xx[:,0],xx[:,1])
    # for i in range ((nbNode)):
    #     # plt.text(xx[i,0], xx[i,1], str(xp[i,1]), fontsize=10, ha='right')
    #     plt.text(xx[i,0], xx[i,1], str(xl[i,1]), fontsize=10, ha='right')
    # plt.axis('equal')
    # plt.show()

    # kk = 0
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.scatter(xx[:,0],xx[:,1])
    # for i in range ((nbNode)):
    #     ax1.text(xx[i,0], xx[i,1], str(xp[i,kk]), fontsize=10, ha='right')
    # plt.axis('equal')
    # ax2.scatter(xx[:,0],xx[:,1])
    # for i in range ((nbNode)):
    #     ax2.text(xx[i,0], xx[i,1], str(xl[i,kk]), fontsize=10, ha='right')
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    # kk = 1
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.scatter(xx[:,0],xx[:,1])
    # for i in range ((nbNode)):
    #     ax1.text(xx[i,0], xx[i,1], str(xp[i,kk]), fontsize=10, ha='right')
    # plt.axis('equal')
    # ax2.scatter(xx[:,0],xx[:,1])
    # for i in range ((nbNode)):
    #     ax2.text(xx[i,0], xx[i,1], str(xl[i,kk]), fontsize=10, ha='right')
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    if not write_debug_files:
        return node_num, element_num, element_list, node_list

    # >>>>>>>>>> Output Nodes' boundaries <<<<<<<<<<
    mask = xp[:, 0] == 1
    xo = np.where(mask)[0]
    np.savetxt("Fix_x1.txt", xo, fmt="%g")

    mask = xp[:, 0] == 9
    xe = np.where(mask)[0]
    np.savetxt("Fix_x2.txt", xe, fmt="%g")

    mask = xp[:, 1] == 1
    yo = np.where(mask)[0]
    np.savetxt("Fix_xy.txt", yo, fmt="%g")

    # >>>>>>>>>> Output Particles' information <<<<<<<<<<
    # ----Block test----
    # npelem = 2
    # # BoundZone = np.array([
    # #     [0.0, 15.0],
    # #     [0.0, 15.0] ])
    # BoundZone = np.array([
    #     [0.0, 300.0],
    #     [0.0, 40.0] ])
    # k = 0  # particle id
    # nbParticle = nbElem * npelem * npelem
    # ParticleInfor = np.zeros((nbParticle, 5))
    # pCoord = np.zeros((1, 2))
    # x = np.linspace(0.0 + 1.0/npelem * 0.5 , 1.0 - 1.0/npelem * 0.5 , npelem)
    # x, y = np.meshgrid(x, x)
    # nodeCoord = np.column_stack((x.ravel(), y.ravel()))
    # for e in range(1, nbElem + 1):
    #     width = baseLength / 2**ElemList[e].level
    #     x = nodeList[ElemList[e].includeNode[0]].coord
    #     for i in range(4):
    #         pCoord = x + nodeCoord[i, :] * width
    #         if BoundZone[0, 0] < pCoord[0] < BoundZone[0, 1] and BoundZone[1, 0] < pCoord[1] < BoundZone[1, 1]:
    #             k += 1
    #             ParticleInfor[k-1, 0:2] = x + nodeCoord[i, :] * width      # particle coord
    #             ParticleInfor[k-1, 2] = width * width / (npelem * npelem)  # particle volume
    #             ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    # print(f'Numbers of Particles: {k}')
    # np.savetxt('particle.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')

    # ----Footing test----
    """npelem = 2
    BoundZone = np.array([
        [0.0, 16.0],
        [0.0, 10.0] ])
    # BoundZone = np.array([
    #     [0.0, 12.0],
    #     [0.0, 8.0] ])
    k = 0  # particle id
    nbParticle = nbElem * npelem * npelem
    ParticleInfor = np.zeros((nbParticle, 5))
    pCoord = np.zeros((1, 2))
    x = np.linspace(0.0 + 1.0/npelem * 0.5 , 1.0 - 1.0/npelem * 0.5 , npelem)
    x, y = np.meshgrid(x, x)
    nodeCoord = np.column_stack((x.ravel(), y.ravel()))
    for e in range(1, nbElem + 1):
        width = baseLength / 2**ElemList[e].level
        x = nodeList[ElemList[e].includeNode[0]].coord
        for i in range(4):
            pCoord = x + nodeCoord[i, :] * width
            if BoundZone[0, 0] < pCoord[0] < BoundZone[0, 1] and BoundZone[1, 0] < pCoord[1] < BoundZone[1, 1]:
                k += 1
                ParticleInfor[k-1, 0:2] = x + nodeCoord[i, :] * width      # particle coord
                ParticleInfor[k-1, 2] = width * width / (npelem * npelem)  # particle volume
                #ParticleInfor[k-1, 2] = width * width / (npelem * npelem) * ParticleInfor[k-1, 0] # particle volume If Axisymmetric
                ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    print(f'Numbers of Particles: {k}')
    np.savetxt('particle_body1.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])
    # plt.axis('equal')
    # plt.show()

    BoundZone = np.array([
        [0.0, 1.0],
        [10.0, 12.0] ])
    # BoundZone = np.array([
    #     [0.0, 1.0],
    #     [8.0, 10.0] ])
    k = 0  # particle id
    nbParticle = nbElem * npelem * npelem
    ParticleInfor = np.zeros((nbParticle, 5))
    pCoord = np.zeros((1, 2))
    x = np.linspace(0.0 + 1.0/npelem * 0.5 , 1.0 - 1.0/npelem * 0.5 , npelem)
    x, y = np.meshgrid(x, x)
    nodeCoord = np.column_stack((x.ravel(), y.ravel()))
    for e in range(1, nbElem + 1):
        width = baseLength / 2**ElemList[e].level
        x = nodeList[ElemList[e].includeNode[0]].coord
        for i in range(4):
            pCoord = x + nodeCoord[i, :] * width
            if BoundZone[0, 0] < pCoord[0] < BoundZone[0, 1] and BoundZone[1, 0] < pCoord[1] < BoundZone[1, 1]:
                k += 1
                ParticleInfor[k-1, 0:2] = x + nodeCoord[i, :] * width      # particle coord
                ParticleInfor[k-1, 2] = width * width / (npelem * npelem)  # particle volume
                #ParticleInfor[k-1, 2] = width * width / (npelem * npelem) * ParticleInfor[k-1, 0] # particle volume If Axisymmetric
                ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    print(f'Numbers of Particles: {k}')
    np.savetxt('particle_body2.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')
    # # check
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])
    # plt.axis('equal')
    # plt.show()"""

    # ----Pile test----
    # npelem = 2
    # BoundZone = np.array([
    #     [0.0, 0.6],
    #     [0.0, 1.5] ])
    # k = 0  # particle id
    # nbParticle = nbElem * npelem * npelem
    # ParticleInfor = np.zeros((nbParticle, 5))
    # pCoord = np.zeros((1, 2))
    # x = np.linspace(0.0 + 1.0/npelem * 0.5 , 1.0 - 1.0/npelem * 0.5 , npelem)
    # x, y = np.meshgrid(x, x)
    # nodeCoord = np.column_stack((x.ravel(), y.ravel()))
    # for e in range(1, nbElem + 1):
    #     width = baseLength / 2**ElemList[e].level
    #     x = nodeList[ElemList[e].includeNode[0]].coord
    #     for i in range(4):
    #         pCoord = x + nodeCoord[i, :] * width
    #         if BoundZone[0, 0] < pCoord[0] < BoundZone[0, 1] and BoundZone[1, 0] < pCoord[1] < BoundZone[1, 1]:
    #             k += 1
    #             ParticleInfor[k-1, 0:2] = x + nodeCoord[i, :] * width      # particle coord
    #             ParticleInfor[k-1, 2] = width * width / (npelem * npelem)  # particle volume
    #             ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    # print(f'Numbers of Particles: {k}')
    # np.savetxt('particle_body1.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])

    # BoundZone = np.array([
    #     [0.0, 0.018],
    #     [1.5, 2.5] ])
    # x0, x1 = BoundZone[0, 0], BoundZone[0, 1]
    # y0, y1 = BoundZone[1, 0], BoundZone[1, 1]
    # k = 0  # particle id
    # nbParticle = nbElem * npelem * npelem
    # ParticleInfor = np.zeros((nbParticle, 5))
    # pCoord = np.zeros((1, 2))
    # x = np.linspace(0.0 + 1.0/npelem * 0.5 , 1.0 - 1.0/npelem * 0.5 , npelem)
    # x, y = np.meshgrid(x, x)
    # nodeCoord = np.column_stack((x.ravel(), y.ravel()))
    # for e in range(1, nbElem + 1):
    #     width = baseLength / 2**ElemList[e].level
    #     x = nodeList[ElemList[e].includeNode[0]].coord
    #     for i in range(4):
    #         pCoord = x + nodeCoord[i, :] * width
    #         xpos, ypos = pCoord[0], pCoord[1]
    #         if x0 < xpos < x1 and y0 + 1.732050808 * (xpos - x0) < ypos < y1:
    #             k += 1
    #             ParticleInfor[k-1, 0:2] = x + nodeCoord[i, :] * width      # particle coord
    #             ParticleInfor[k-1, 2] = width * width / (npelem * npelem)  # particle volume
    #             ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    #  ----Opend-ended Pile simulation----
    npelem = 2
    # BoundZone = np.array([
    #     [0.0, 59.2],
    #     [0.0, 48.0] ])
    # BoundZone = np.array([
    #     [0.0, 30.0],
    #     [0.0, 40.0] ])
    bound_zone = np.array([[0.0, 114.4], [0.0, 90.0]])
    k = 0  # particle id
    nb_particle = element_num * npelem * npelem
    particle_information = np.zeros((nb_particle, 5))
    p_coord = np.zeros((1, 2))
    x = np.linspace(0.0 + 1.0 / npelem * 0.5, 1.0 - 1.0 / npelem * 0.5, npelem)
    x, y = np.meshgrid(x, x)
    node_coord = np.column_stack((x.ravel(), y.ravel()))
    for e in range(1, element_num + 1):
        width = baseLength / 2 ** element_list[e].level
        x = node_list[element_list[e].includeNode[0]].coord
        for i in range(4):
            p_coord = x + node_coord[i, :] * width
            if bound_zone[0, 0] < p_coord[0] < bound_zone[0, 1] and bound_zone[1, 0] < p_coord[1] < bound_zone[1, 1]:
                k += 1
                particle_information[k - 1, 0:2] = x + node_coord[i, :] * width  # particle coord
                particle_information[k - 1, 2] = width * width / (npelem * npelem)  # particle volume
                particle_information[k - 1, 3:5] = np.array(
                    [width / npelem * 0.5, width / npelem * 0.5]
                )  # particle psize

    print(f"Numbers of Particles: {k}")
    np.savetxt("particle_body1.txt", particle_information[:k, :], fmt="%.6f", delimiter=" ")
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])

    # BoundZone = np.array([
    #     [5.0, 5.2],
    #     [48.0, 86.0] ])
    # BoundZone = np.array([
    #     [2.4, 2.6],
    #     [40.0, 60.0] ])
    bound_zone = np.array([[10.2, 10.4], [90.0, 160.0]])
    dL = 0.05 / 2.0
    dp = dL * 0.5
    x0, x1 = bound_zone[0, 0], bound_zone[0, 1]
    y0, y1 = bound_zone[1, 0], bound_zone[1, 1]
    k = 0  # particle id
    numx = int((x1 - x0) / dp)
    numy = int((y1 - y0) / dp)
    nb_particle = element_num * npelem * npelem
    particle_information = np.zeros((nb_particle, 5))
    for iy in range(numy):
        for ix in range(numx):
            p_coord[0] = x0 + dp * 0.5 + dp * ix
            p_coord[1] = y0 + dp * 0.5 + dp * iy
            # if (pCoord[1] < 48.1 and pCoord[0] < 5.075 and -(pCoord[1]-48.0)/(pCoord[0]-5.075) < 2.) or (pCoord[1] < 48.1 and pCoord[0] > 5.125 and (pCoord[1]-48.0)/(pCoord[0]-5.125) < 2.):
            # if (pCoord[1] < 40.1 and pCoord[0] < 2.475 and -(pCoord[1]-40.0)/(pCoord[0]-2.475) < 2.) or (pCoord[1] < 40.1 and pCoord[0] > 2.525 and (pCoord[1]-40.0)/(pCoord[0]-2.525) < 2.):
            if (p_coord[1] < 90.1 and p_coord[0] < 10.275 and -(p_coord[1] - 90.0) / (p_coord[0] - 10.275) < 2.0) or (
                p_coord[1] < 90.1 and p_coord[0] > 10.325 and (p_coord[1] - 90.0) / (p_coord[0] - 10.325) < 2.0
            ):
                None  # 5.075<----->5.125
            else:
                k += 1
                particle_information[k - 1, 0] = x0 + dp * 0.5 + dp * ix  # particle coord
                particle_information[k - 1, 1] = y0 + dp * 0.5 + dp * iy  # particle coord
                particle_information[k - 1, 2] = dp * dp  # particle volume
                particle_information[k - 1, 3:5] = np.array([dp * 0.5, dp * 0.5])  # particle psize

    print(f"Numbers of Particles: {k}")
    np.savetxt("particle_body2.txt", particle_information[:k, :], fmt="%.6f", delimiter=" ")
    # check
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])
    # plt.axis('equal')
    # plt.show()

    return node_num, element_num, element_list, node_list


@ti.func
def adaptive_list_count(count: ti.template()):
    value = 0
    if ti.static(hasattr(count, "snode")):
        value = count[None]
    else:
        value = count
    return value


def build_bridging_node_weights(refined_cell, coarse_cnum, bridging_cells):
    coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
    coarse_gnum = coarse_cnum + 1
    dimension = coarse_cnum.size
    refined = np.asarray(refined_cell, dtype=np.uint8).reshape(
        tuple(coarse_cnum),
        order="F",
    )
    node_has_refined_cell = np.zeros(tuple(coarse_gnum), dtype=bool)
    node_has_coarse_cell = np.zeros(tuple(coarse_gnum), dtype=bool)

    for cell in np.ndindex(*coarse_cnum):
        target = node_has_refined_cell if refined[cell] else node_has_coarse_cell
        for offset in product((0, 1), repeat=dimension):
            node = tuple(cell[d] + offset[d] for d in range(dimension))
            target[node] = True

    interface = node_has_refined_cell & node_has_coarse_cell
    distance = np.full(tuple(coarse_gnum), -1, dtype=np.int32)
    frontier = deque()
    for node in zip(*np.nonzero(interface)):
        distance[node] = 0
        frontier.append(node)

    while frontier:
        node = frontier.popleft()
        if distance[node] >= bridging_cells:
            continue
        for axis in range(dimension):
            for direction in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += direction
                if not 0 <= neighbor[axis] < coarse_gnum[axis]:
                    continue
                neighbor = tuple(neighbor)
                if node_has_refined_cell[neighbor] and distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    frontier.append(neighbor)

    weights = np.zeros(tuple(coarse_gnum), dtype=np.float64)
    in_bridge = (distance >= 0) & (distance <= bridging_cells)
    weights[in_bridge] = 1.0 - distance[in_bridge] / float(bridging_cells)
    return np.ascontiguousarray(weights.flatten(order="F"))


def build_hanging_constraint_table(
    refined_cell,
    coarse_cnum,
    fine_gnum,
    logical_to_compact,
    compact_to_logical,
    active_compact_nodes,
    refinement_ratio=2,
    max_level=1,
):
    coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
    fine_gnum = np.asarray(fine_gnum, dtype=np.int32)
    refined_cell = np.asarray(refined_cell, dtype=np.uint8).reshape(
        tuple(coarse_cnum),
        order="F",
    )
    logical_to_compact = np.asarray(logical_to_compact, dtype=np.int32)
    compact_to_logical = np.asarray(compact_to_logical, dtype=np.int32)
    dimension = coarse_cnum.size
    corner_count = 1 << dimension
    finest_ratio = int(refinement_ratio) ** int(max_level)

    slave_ids = []
    master_ids = []
    master_weights = []
    touched_nodes = set()

    for node_id in range(active_compact_nodes):
        logical_id = int(compact_to_logical[node_id])
        if logical_id < 0:
            continue
        fine_node = np.asarray(
            np.unravel_index(logical_id, tuple(fine_gnum), order="F"),
            dtype=np.int32,
        )

        native_level = 0
        for level in range(1, max_level + 1):
            stride = finest_ratio // (refinement_ratio**level)
            if np.all(fine_node % stride == 0):
                native_level = level
        if native_level == 0:
            continue

        adjacent_levels = []
        has_adjacent_cell = False
        base_cell = fine_node // finest_ratio
        for offset in product((0, 1), repeat=dimension):
            adjacent_cell = base_cell.copy()
            for d in range(dimension):
                if fine_node[d] % finest_ratio == 0:
                    adjacent_cell[d] += offset[d] - 1
            if np.all(adjacent_cell >= 0) and np.all(adjacent_cell < coarse_cnum):
                has_adjacent_cell = True
                adjacent_levels.append(int(refined_cell[tuple(adjacent_cell)]))

        if not has_adjacent_cell:
            continue
        master_level = min(adjacent_levels)
        if master_level >= native_level:
            continue

        master_divisions = refinement_ratio**master_level
        master_stride = finest_ratio // master_divisions
        master_cell = np.minimum(fine_node // master_stride, coarse_cnum * master_divisions - 1)
        local_coord = (fine_node.astype(np.float64) - master_stride * master_cell.astype(np.float64)) / float(
            master_stride
        )
        ids = np.full(corner_count, -1, dtype=np.int32)
        weights = np.zeros(corner_count, dtype=np.float64)
        for corner, offset in enumerate(product((0, 1), repeat=dimension)):
            weight = 1.0
            for d in range(dimension):
                weight *= local_coord[d] if offset[d] else 1.0 - local_coord[d]
            weights[corner] = weight
            if weight > 0.0:
                master_node = master_stride * (master_cell + np.asarray(offset, dtype=np.int32))
                master_logical_id = np.ravel_multi_index(
                    tuple(master_node),
                    tuple(fine_gnum),
                    order="F",
                )
                master_id = int(logical_to_compact[master_logical_id])
                assert master_id >= 0, "AdaptiveGrid coarse node map is incomplete"
                ids[corner] = master_id
                touched_nodes.add(master_id)

        nonzero_master_ids = ids[weights > 0.0]
        if nonzero_master_ids.size == 1 and int(nonzero_master_ids[0]) == node_id:
            continue

        slave_ids.append(node_id)
        master_ids.append(ids)
        master_weights.append(weights)
        touched_nodes.add(node_id)

    # A master on a coarse/fine edge can itself be constrained at another level.
    constraints = {
        slave: {int(master): float(weight) for master, weight in zip(ids, weights) if weight > 0.0}
        for slave, ids, weights in zip(slave_ids, master_ids, master_weights)
    }
    resolved = {}
    visiting = set()

    def independent_masters(node_id):
        if node_id not in constraints:
            return {node_id: 1.0}
        if node_id not in resolved:
            if node_id in visiting:
                raise SimulationStateError("AdaptiveGrid hanging constraints contain a cycle")
            visiting.add(node_id)
            expanded = {}
            for master, weight in constraints[node_id].items():
                for independent, fraction in independent_masters(master).items():
                    expanded[independent] = expanded.get(independent, 0.0) + weight * fraction
            visiting.remove(node_id)
            resolved[node_id] = expanded
        return resolved[node_id]

    for index, slave in enumerate(slave_ids):
        expanded = independent_masters(slave)
        if len(expanded) > corner_count:
            raise CapacityError("AdaptiveGrid expanded hanging constraint exceeds master capacity")
        master_ids[index].fill(-1)
        master_weights[index].fill(0.0)
        for slot, (master, weight) in enumerate(sorted(expanded.items())):
            master_ids[index][slot] = master
            master_weights[index][slot] = weight

    return (
        np.ascontiguousarray(slave_ids, dtype=np.int32),
        np.ascontiguousarray(master_ids, dtype=np.int32).reshape(-1, corner_count),
        np.ascontiguousarray(master_weights, dtype=np.float64).reshape(
            -1,
            corner_count,
        ),
        np.ascontiguousarray(sorted(touched_nodes), dtype=np.int32),
    )
