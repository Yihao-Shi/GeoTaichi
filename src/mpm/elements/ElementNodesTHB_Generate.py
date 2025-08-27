import numpy as np
import matplotlib.pyplot as plt

# Define the Element class
class Element:
    def __init__(self):
        self.divide = False
        self.childElem = []
        self.neighbor = [0] * 4
        self.includeNode = [0] * 8
        self.influenNode = [0] * 25
        self.nbInfNode = 0
        self.level = -1
        self.center = [0.0] * 2

# Define the Node class
class Node:
    def __init__(self):
        self.coord = [0.0] * 2
        self.level = [-1] * 2
        self.Ntype = [3] * 2

# Initialize global variables
ElemList = []
nodeList = []
nbElem = 0
nbNode = 0
level = 0
npelem = 2
nbParticle = 0
ParticleInfor = []

def ElemNodesGen(baseLength, modelRange, level, refineBound):
    global ElemList, nodeList, nbElem, nbNode, ParticleInfor
    
    Nx = int((modelRange[0][1] - modelRange[0][0]) / baseLength + 0.5) + 1
    Ny = int((modelRange[1][1] - modelRange[1][0]) / baseLength + 0.5) + 1

    nodeList = [Node() for _ in range(2000000)]
    ElemList = [Element() for _ in range(2000001)]

    for i in range(1, 2000001):
        ElemList[i].childElem = [0] * level

    for iy in range(1, Ny + 1):
        for ix in range(1, Nx + 1):
            i = (iy - 1) * Nx + ix
            nodeList[i].coord[0] = (ix - 1) * baseLength + modelRange[0][0]
            nodeList[i].coord[1] = (iy - 1) * baseLength + modelRange[1][0]
            nodeList[i].level = [0] * 2

    nbNode = Nx * Ny

    for iy in range(1, Ny):
        for ix in range(1, Nx):
            i = (iy - 1) * (Nx - 1) + ix
            Node1 = (iy - 1) * Nx + ix
            ElemList[i].includeNode[0] = Node1
            ElemList[i].includeNode[1] = Node1 + 1
            ElemList[i].includeNode[2] = Node1 + 1 + Nx
            ElemList[i].includeNode[3] = Node1 + Nx

            ElemList[i].center[0] = 0.5 * (nodeList[Node1].coord[0] + nodeList[Node1 +  1].coord[0])
            ElemList[i].center[1] = 0.5 * (nodeList[Node1].coord[1] + nodeList[Node1 + Nx].coord[1])

            if iy > 1:
                ElemList[i].neighbor[0] = i + 1 - Nx
            if ix < Nx - 1:
                ElemList[i].neighbor[1] = i + 1
            if iy < Ny - 1:
                ElemList[i].neighbor[2] = i + Nx - 1
            if ix > 1:
                ElemList[i].neighbor[3] = i - 1

            ElemList[i].level = 0

    nbElem = (Nx - 1) * (Ny - 1)

    print(f'Numbers of Element and Node: {nbElem}, {nbNode}')

    for l in range(1, level + 1):
        L = baseLength / (2 ** l)
        nodeCoord = [[L, 0.0], [2.0*L , L], [L, 2.0*L], [0.0, L]]

        for i in range(1, nbElem + 1):
            if ElemList[i].level != (l - 1):
                continue
            elemCenter = ElemList[i].center

            if (refineBound[l-1][0][0] <= elemCenter[0] <= refineBound[l-1][0][1] and
                refineBound[l-1][1][0] <= elemCenter[1] <= refineBound[l-1][1][1]):
                ElemList[i].divide = True

            if ElemList[i].divide:
                nbNewnode = 0
                for j in range(4, 8):
                    if ElemList[i].includeNode[j] < 1:
                        nbNewnode += 1
                        ElemList[i].includeNode[j] = nbNode + nbNewnode
                        nodeList[nbNode + nbNewnode].coord = [nodeList[ElemList[i].includeNode[0]].coord[k] + nodeCoord[j - 4][k] for k in range(2)]

                nbNewnode += 1
                nodeList[nbNode + nbNewnode].coord = [nodeList[ElemList[i].includeNode[0]].coord[k] + baseLength / (2 ** l) for k in range(2)]

                ElemList[i].center = [nodeList[ElemList[i].includeNode[0]].coord[k] + 0.5 * baseLength / (2 ** l) for k in range(2)]
                ElemList[nbElem + 1].center = [nodeList[ElemList[i].includeNode[0]].coord[0] + 1.5 * baseLength / (2 ** l), nodeList[ElemList[i].includeNode[0]].coord[1] + 0.5 * baseLength / (2 ** l)]
                ElemList[nbElem + 2].center = [nodeList[ElemList[i].includeNode[0]].coord[k] + 1.5 * baseLength / (2 ** l) for k in range(2)]
                ElemList[nbElem + 3].center = [nodeList[ElemList[i].includeNode[0]].coord[0] + 0.5 * baseLength / (2 ** l), nodeList[ElemList[i].includeNode[0]].coord[1] + 1.5 * baseLength / (2 ** l)]

                ElemList[nbElem + 1].includeNode[0] = ElemList[i].includeNode[4]
                ElemList[nbElem + 1].includeNode[1] = ElemList[i].includeNode[1]
                ElemList[nbElem + 1].includeNode[2] = ElemList[i].includeNode[5]
                ElemList[nbElem + 1].includeNode[3] = nbNode + nbNewnode

                ElemList[nbElem + 2].includeNode[0] = nbNode + nbNewnode
                ElemList[nbElem + 2].includeNode[1] = ElemList[i].includeNode[5]
                ElemList[nbElem + 2].includeNode[2] = ElemList[i].includeNode[2]
                ElemList[nbElem + 2].includeNode[3] = ElemList[i].includeNode[6]

                ElemList[nbElem + 3].includeNode[0] = ElemList[i].includeNode[7]
                ElemList[nbElem + 3].includeNode[1] = nbNode + nbNewnode
                ElemList[nbElem + 3].includeNode[2] = ElemList[i].includeNode[6]
                ElemList[nbElem + 3].includeNode[3] = ElemList[i].includeNode[3]

                elnode = [6, 7, 4, 5]
                for j in range(4):
                    e = ElemList[i].neighbor[j]
                    if e == 0 or ElemList[e].divide:
                        continue
                    ElemList[e].includeNode[elnode[j]] = ElemList[i].includeNode[j + 4]

                e = ElemList[i].neighbor[0]
                if e != 0:
                    ElemList[nbElem + 1].neighbor[0] = e
                    if ElemList[e].divide:
                        e1 = e
                        e2 = ElemList[e1].childElem[l-1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        ElemList[i].neighbor[0] = e4
                        ElemList[e4].neighbor[2] = i
                        ElemList[nbElem + 1].neighbor[0] = e3
                        ElemList[e3].neighbor[2] = nbElem + 1

                e = ElemList[i].neighbor[1]
                if e != 0:
                    ElemList[nbElem + 1].neighbor[1] = e
                    ElemList[nbElem + 2].neighbor[1] = e
                    if ElemList[e].divide:
                        e1 = e
                        e2 = ElemList[e1].childElem[l-1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        ElemList[nbElem + 1].neighbor[1] = e1
                        ElemList[e1].neighbor[3] = nbElem + 1
                        ElemList[nbElem + 2].neighbor[1] = e4
                        ElemList[e4].neighbor[3] = nbElem + 2

                e = ElemList[i].neighbor[2]
                if e != 0:
                    ElemList[nbElem + 2].neighbor[2] = e
                    ElemList[nbElem + 3].neighbor[2] = e
                    if ElemList[e].divide:
                        e1 = e
                        e2 = ElemList[e1].childElem[l-1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        ElemList[nbElem + 2].neighbor[2] = e2
                        ElemList[e2].neighbor[0] = nbElem + 2
                        ElemList[nbElem + 3].neighbor[2] = e1
                        ElemList[e1].neighbor[0] = nbElem + 3

                e = ElemList[i].neighbor[3]
                if e != 0:
                    ElemList[nbElem + 3].neighbor[3] = e
                    if ElemList[e].divide:
                        e1 = e
                        e2 = ElemList[e1].childElem[l-1]
                        e3 = e2 + 1
                        e4 = e2 + 2
                        ElemList[i].neighbor[3] = e2
                        ElemList[e2].neighbor[1] = i
                        ElemList[nbElem + 3].neighbor[3] = e3
                        ElemList[e3].neighbor[1] = nbElem + 3

                ElemList[i].neighbor[1] = nbElem + 1
                ElemList[i].neighbor[2] = nbElem + 3
                ElemList[nbElem + 1].neighbor[2] = nbElem + 2
                ElemList[nbElem + 1].neighbor[3] = i
                ElemList[nbElem + 2].neighbor[0] = nbElem + 1
                ElemList[nbElem + 2].neighbor[3] = nbElem + 3
                ElemList[nbElem + 3].neighbor[0] = i
                ElemList[nbElem + 3].neighbor[1] = nbElem + 2

                ElemList[i].includeNode[1] = ElemList[i].includeNode[4]
                ElemList[i].includeNode[2] = nbNode + nbNewnode
                ElemList[i].includeNode[3] = ElemList[i].includeNode[7]
                ElemList[i].includeNode[4:8] = [0] * 4

                ElemList[i].childElem[l-1] = nbElem + 1

                ElemList[i].level = l
                for j in range(1, 4):
                    ElemList[nbElem + j].level = l

                nbNode += nbNewnode
                nbElem += 3

        for elem in ElemList:
            elem.divide = False

    for e in range(1, nbElem + 1):
        elem = ElemList[e]
        Level = elem.level
        for i in range(4):
            inode = elem.includeNode[i]
            nodeList[inode].level = [Level] * 2
            nodeList[inode].Ntype = [3] * 2

    for e in range(1, nbElem + 1):
        elem = ElemList[e]
        Level = elem.level
        neiElem = elem.neighbor

        e1 = neiElem[0]
        l1 = ElemList[e1].level if e1 != 0 else -1
        e2 = neiElem[1]
        l2 = ElemList[e2].level if e2 != 0 else -1
        e3 = neiElem[2]
        l3 = ElemList[e3].level if e3 != 0 else -1
        e4 = neiElem[3]
        l4 = ElemList[e4].level if e4 != 0 else -1

        if 0 in neiElem:
            if neiElem[3] == 0:
                if Level < l3 and l3 >0:
                    None
                elif Level < l1 and l1 >=0:
                    None
                else:
                    nodeList[elem.includeNode[0]].Ntype[0] = 1
                    nodeList[elem.includeNode[3]].Ntype[0] = 1
                    nodeList[elem.includeNode[1]].Ntype[0] = 2
                    nodeList[elem.includeNode[2]].Ntype[0] = 2
                if Level > l1 >= 0:
                    for k in range(0,2):
                        nodeList[elem.includeNode[k]].Ntype[1] = 4
                    for k in range(2,4):
                        nodeList[elem.includeNode[k]].Ntype[1] = 5
                    for k in range(0,4):
                        nodeList[elem.includeNode[k]].level[1] = Level - 1
                    nodeList[elem.includeNode[0]].level[0] = Level
                elif Level > l3 >= 0:
                    for k in range(2,4):
                        nodeList[elem.includeNode[k]].Ntype[1] = 7
                    for k in range(0,2):
                        nodeList[elem.includeNode[k]].Ntype[1] = 6
                    for k in range(0,4):
                        nodeList[elem.includeNode[k]].level[1] = Level - 1
                    nodeList[ElemList[e2].includeNode[2]].Ntype[0] = 3
            if neiElem[1] == 0:
                for k in range(1,3):
                    nodeList[elem.includeNode[k]].Ntype[0] = 9
                nodeList[elem.includeNode[0]].Ntype[0] = 8
                nodeList[elem.includeNode[3]].Ntype[0] = 8
            if neiElem[0] == 0:
                if Level < l2 and l2 >0:
                    None
                elif Level < l4 and l4 >=0:
                    None
                else:
                    nodeList[elem.includeNode[0]].Ntype[1] = 1
                    nodeList[elem.includeNode[1]].Ntype[1] = 1
                    nodeList[elem.includeNode[2]].Ntype[1] = 2
                    nodeList[elem.includeNode[3]].Ntype[1] = 2
                if Level > l4 >= 0:
                    nodeList[elem.includeNode[0]].Ntype[0] = 4
                    nodeList[elem.includeNode[3]].Ntype[0] = 4
                    nodeList[elem.includeNode[1]].Ntype[0] = 5
                    nodeList[elem.includeNode[2]].Ntype[0] = 5
                    for k in range(0,4):
                        nodeList[elem.includeNode[k]].level[0] = Level - 1
                    nodeList[ElemList[e3].includeNode[3]].Ntype[1] = 3
                elif Level > l2 >= 0:
                    nodeList[elem.includeNode[1]].Ntype[0] = 7
                    nodeList[elem.includeNode[2]].Ntype[0] = 7
                    nodeList[elem.includeNode[0]].Ntype[0] = 6
                    nodeList[elem.includeNode[3]].Ntype[0] = 6
                    for k in range(0,4):
                        nodeList[elem.includeNode[k]].level[0] = Level - 1
                    nodeList[ElemList[e3].includeNode[2]].Ntype[1] = 3
            if neiElem[2] == 0:
                for k in range(2,4):
                    nodeList[elem.includeNode[k]].Ntype[1] = 9
                for k in range(0,2):
                    nodeList[elem.includeNode[k]].Ntype[1] = 8
        else:
            if Level > l1:
                for k in range(0,2):
                    nodeList[elem.includeNode[k]].Ntype[1] = 4
                for k in range(2,4):
                    nodeList[elem.includeNode[k]].Ntype[1] = 5
                for k in range(0,4):
                    nodeList[elem.includeNode[k]].level[1] = Level - 1
            elif Level > l3:
                for k in range(2,4):
                    nodeList[elem.includeNode[k]].Ntype[1] = 7
                for k in range(0,2):
                    nodeList[elem.includeNode[k]].Ntype[1] = 6
                for k in range(0,4):
                    nodeList[elem.includeNode[k]].level[1] = Level - 1
            if Level > l4:
                nodeList[elem.includeNode[0]].Ntype[0] = 4
                nodeList[elem.includeNode[3]].Ntype[0] = 4
                for k in range(1,3):
                    nodeList[elem.includeNode[k]].Ntype[0] = 5
                for k in range(0,4):
                    nodeList[elem.includeNode[k]].level[0] = Level - 1
            elif Level > l2:
                nodeList[elem.includeNode[0]].Ntype[0] = 6
                nodeList[elem.includeNode[3]].Ntype[0] = 6
                for k in range(1,3):
                    nodeList[elem.includeNode[k]].Ntype[0] = 7
                for k in range(0,4):
                    nodeList[elem.includeNode[k]].level[0] = Level - 1

    # >>>>>>>>>> Assign elements' influenced nodes <<<<<<<<<<
    nodeInflZone = [[0., 0.], [0., 0.]]
    threshold = baseLength * 1.e-3
    for i in range(1, nbNode + 1):
        x = nodeList[i].coord
        for j in range(2):
            k = nodeList[i].level[j]
            width = baseLength / (2 ** k)
            nodeInflZone[j][0] = x[j] - 2.0 * width
            nodeInflZone[j][1] = x[j] + 2.0 * width
            nodetype = nodeList[i].Ntype

            if nodetype[j] == 4:
                nodeInflZone[j][0] = x[j] - 2.0 * width
                nodeInflZone[j][1] = x[j] + 1.5 * width
            elif nodetype[j] == 5:
                nodeInflZone[j][0] = x[j] - 1.5 * width
                nodeInflZone[j][1] = x[j] + 1.0 * width
            elif nodetype[j] == 6:
                nodeInflZone[j][0] = x[j] - 1.0 * width
                nodeInflZone[j][1] = x[j] + 1.5 * width
            elif nodetype[j] == 7:
                nodeInflZone[j][0] = x[j] - 1.5 * width
                nodeInflZone[j][1] = x[j] + 2.0 * width

        for e in range(1, nbElem + 1):
            if (nodeInflZone[0][0]-threshold <= ElemList[e].center[0] <= nodeInflZone[0][1]+threshold and
                nodeInflZone[1][0]-threshold <= ElemList[e].center[1] <= nodeInflZone[1][1]+threshold):
                ElemList[e].nbInfNode += 1
                ElemList[e].influenNode[ElemList[e].nbInfNode - 1] = i

    # check node information
    xx = np.zeros((nbNode,2))
    xp = np.zeros((nbNode,2), dtype=int)
    xl = np.zeros((nbNode,2), dtype=int)
    for i in range(1, nbNode + 1):
        xx[i-1][:] = nodeList[i].coord[:]
        xp[i-1][:] = nodeList[i].Ntype[:]
        xl[i-1][:] = nodeList[i].level[:]   
        
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

    # >>>>>>>>>> Output Nodes' boundaries <<<<<<<<<<
    mask = xp[:, 0] == 1
    xo = np.where(mask)[0]
    np.savetxt('Fix_x1.txt', xo, fmt='%g')

    mask = xp[:, 0] == 9
    xe = np.where(mask)[0]
    np.savetxt('Fix_x2.txt', xe, fmt='%g')
    
    mask = xp[:, 1] == 1
    yo = np.where(mask)[0]
    np.savetxt('Fix_xy.txt', yo, fmt='%g')


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
    '''npelem = 2
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
    # plt.show()'''

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
    BoundZone = np.array([
        [0.0, 114.4],
        [0.0, 90.0] ])
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
                ParticleInfor[k-1, 3:5] = np.array([width/npelem * 0.5, width/npelem * 0.5])         # particle psize

    print(f'Numbers of Particles: {k}')
    np.savetxt('particle_body1.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])

    # BoundZone = np.array([
    #     [5.0, 5.2],
    #     [48.0, 86.0] ])
    # BoundZone = np.array([
    #     [2.4, 2.6],
    #     [40.0, 60.0] ])
    BoundZone = np.array([
        [10.2, 10.4],
        [90.0, 160.0] ])
    dL = 0.05/2.
    dp = dL * 0.5
    x0, x1 = BoundZone[0, 0], BoundZone[0, 1]
    y0, y1 = BoundZone[1, 0], BoundZone[1, 1]
    k = 0  # particle id
    numx = int((x1 - x0) / dp)
    numy = int((y1 - y0) / dp)
    nbParticle = nbElem * npelem * npelem
    ParticleInfor = np.zeros((nbParticle, 5))
    for iy in range(numy):
        for ix in range(numx):
            pCoord[0] = x0 + dp * 0.5 + dp * ix
            pCoord[1] = y0 + dp * 0.5 + dp * iy
            # if (pCoord[1] < 48.1 and pCoord[0] < 5.075 and -(pCoord[1]-48.0)/(pCoord[0]-5.075) < 2.) or (pCoord[1] < 48.1 and pCoord[0] > 5.125 and (pCoord[1]-48.0)/(pCoord[0]-5.125) < 2.):
            # if (pCoord[1] < 40.1 and pCoord[0] < 2.475 and -(pCoord[1]-40.0)/(pCoord[0]-2.475) < 2.) or (pCoord[1] < 40.1 and pCoord[0] > 2.525 and (pCoord[1]-40.0)/(pCoord[0]-2.525) < 2.):
            if (pCoord[1] < 90.1 and pCoord[0] < 10.275 and -(pCoord[1]-90.0)/(pCoord[0]-10.275) < 2.) or (pCoord[1] < 90.1 and pCoord[0] > 10.325 and (pCoord[1]-90.0)/(pCoord[0]-10.325) < 2.):
                None  # 5.075<----->5.125
            else:
                k += 1
                ParticleInfor[k-1, 0] = x0 + dp * 0.5 + dp * ix      # particle coord
                ParticleInfor[k-1, 1] = y0 + dp * 0.5 + dp * iy      # particle coord
                ParticleInfor[k-1, 2] = dp * dp     # particle volume
                ParticleInfor[k-1, 3:5] = np.array([dp * 0.5, dp * 0.5])         # particle psize                

    print(f'Numbers of Particles: {k}')
    np.savetxt('particle_body2.txt', ParticleInfor[:k,:], fmt='%.6f', delimiter=' ')
    # check
    # plt.figure(2)
    # plt.scatter(ParticleInfor[:k,0],ParticleInfor[:k,1])
    # plt.axis('equal')
    # plt.show()

    return nbNode, nbElem, ElemList, nodeList






