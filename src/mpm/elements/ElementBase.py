from src.utils.TypeDefination import vec3f, vec3i


class ElementBase(object):
    def __init__(self) -> None:
        self.multi_grids = None
        self.resolution = 0.1

    def create_nodes(self, sims, grid_size):
        raise NotImplementedError

    def calc_volume(self):
        raise NotImplementedError

    def calc_total_particle(self, npic):
        raise NotImplementedError

    def calc_particle_size(self, npic):
        raise NotImplementedError
    
    def get_total_cell_number(self):
        raise NotImplementedError
    
    def activate_cell(self, sims):
        raise NotImplementedError
    
    def set_up_cell_active_flag(self, fb):
        raise NotImplementedError
    
    def update_particle_in_cell(self, particleNum, particle):
        raise NotImplementedError
    
    def activate_gauss_cell(self, sims):
        raise NotImplementedError