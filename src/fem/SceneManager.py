"""Scene storage for the FEM facade."""


class FEMScene:
    def __init__(self):
        self.mesh = None
        self.material = None
        self.dirichlet = None
        self.neumann = None
        self.contact = None
        self.soft_particle_contact = None
        self.cloth_energies = []

    def add_mesh(self, mesh):
        self.mesh = mesh

    def add_material(self, material):
        self.material = material

    def add_boundary_condition(self, dirichlet=None, neumann=None):
        if dirichlet is not None:
            self.dirichlet = dirichlet
        if neumann is not None:
            self.neumann = neumann

    def add_contact(self, contact):
        self.contact = contact

    def add_soft_particle_contact(self, contact):
        self.soft_particle_contact = contact

    def add_cloth_energy(self, energy):
        self.cloth_energies.append(dict(energy))


__all__ = ["FEMScene"]
