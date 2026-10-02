class IGAScene:
    def __init__(self):
        self.primitives = None
        self.rest_shape = None
        self.dirichlet = None
        self.neumann = None
        self.material = {}
        self.element = {}

    def add_primitives(self, primitives, rest_shape=None):
        self.primitives = primitives
        self.rest_shape = rest_shape

    def add_boundary_condition(self, dirichlet=None, neumann=None):
        self.dirichlet = dirichlet
        self.neumann = neumann

    def add_material(self, **kwargs):
        self.material.update(kwargs)

    def add_element(self, degree):
        self.element["degree"] = degree
