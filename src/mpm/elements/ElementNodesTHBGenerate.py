THB_MAX_INFLUENCE_NODES = 32


class Element:
    def __init__(self):
        self.divide = False
        self.childElem = []
        self.neighbor = [0] * 4
        self.includeNode = [0] * 8
        self.influenNode = [0] * THB_MAX_INFLUENCE_NODES
        self.nbInfNode = 0
        self.level = -1
        self.center = [0.0] * 2


class Node:
    def __init__(self):
        self.coord = [0.0] * 2
        self.level = [-1] * 2
        self.Ntype = [3] * 2


element_list = []

node_list = []

element_num = 0

node_num = 0

level = 0

npelem = 2

particle_num = 0

particle_information = []
