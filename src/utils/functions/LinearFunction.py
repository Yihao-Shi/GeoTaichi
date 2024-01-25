import taichi as ti

from src.utils.ObjectIO import DictIO


@ti.dataclass
class Function1D:
    beg_time: float
    end_time: float
    gradient: float
    offset: float

    def set_value(self, to_beg, to_end, to_beg_val, to_end_val):
        self.beg_time = to_beg
        self.end_time = to_end
        self.gradient = (to_end_val - to_beg_val) / (to_end - to_beg)
        self.offset = to_end_val - self.gradient * to_end

    @ti.func
    def _get_start_value(self):
        return self.gradient * self.beg_time + self.offset

    @ti.func
    def _get_final_value(self):
        return self.gradient * self.end_time + self.offset

    @ti.func
    def _get_value(self, t):
        return self.gradient * t + self.offset


@ti.data_oriented
class LinearFunction:
    def __init__(self):
        self.i_interval = 0
        self.current_interval = 0
        self.function = None

    def set_function(self, functions):
        self.i_interval = len(functions) - 1
        if self.i_interval < 1:
            raise ValueError("The Length of Function is not enough!")

        self.function = Function1D.field(shape=self.i_interval)
        
        start_time = DictIO.GetEssential(functions[0], "TimeStamp")
        start_value = DictIO.GetEssential(functions[0], "Value")
        for i in range(1, self.i_interval):
            end_time = DictIO.GetEssential(functions[i], "TimeStamp")
            end_value = DictIO.GetEssential(functions[i], "Value")
            if end_time <= start_time:
                raise ValueError("The interval time should be larger than zero")
            self.function[i - 1].set_value(start_time, end_time, start_value, end_value)

            start_time = end_time
            start_value =end_value          

    @ti.func
    def get_interval_value(self, curr_time):
        curr_value = self.function[self.i_interval - 1]._get_final_value()
        if self.current_interval < self.i_interval:
            for i in range(self.current_interval, self.i_interval):
                if self.function[i].beg_time <= curr_time < self.function[i].end_time:
                    self.current_interval = i
                    curr_value = self.function[self.current_interval]._get_value(curr_time)
                    break
        return curr_value
