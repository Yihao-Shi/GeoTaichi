import taichi as ti



class ConstantFunction:
    def __init__(self) -> None:
        self.value = 0.

    def set_function(self, value):
        self.value = value

    def get_interval_value(self, curr_time):
        return self.value