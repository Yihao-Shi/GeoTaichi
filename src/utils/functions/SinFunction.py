import taichi as ti

from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


@ti.dataclass
class Function:
    beg_time: float
    end_time: float
    iperiod: float
    phase: float
    magnitude: float

    def set_value(self, to_beg, to_end, period, phase, magnitude):
        self.beg_time = to_beg
        self.end_time = to_end
        self.iperiod = 1. / period
        self.phase = phase
        self.magnitude = magnitude

    @ti.func
    def _get_start_value(self):
        return self.magnitude * ti.sin(2 * PI * self.iperiod * self.beg_time + self.phase)

    @ti.func
    def _get_final_value(self):
        return self.magnitude * ti.sin(2 * PI * self.iperiod * self.end_time + self.phase)

    @ti.func
    def _get_value(self, t):
        return self.magnitude * ti.sin(2 * PI * self.iperiod * t + self.phase)


@ti.data_oriented
class SinFunction:
    def __init__(self):
        self.i_interval = 0
        self.current_interval = 0
        self.function = None

    def set_function(self, functions):
        self.i_interval = len(functions)
        if self.i_interval < 1:
            raise ValueError("The Length of Function is not enough!")

        self.function = Function.field(shape=self.i_interval)
        for i in range(self.i_interval):
            start_time = DictIO.GetEssential(functions[0], "StartTime")
            end_time = DictIO.GetEssential(functions[i], "EndTime")
            period = DictIO.GetEssential(functions[i], "Period")
            phase = DictIO.GetEssential(functions[i], "Phase") / 180 * PI
            magnitude = DictIO.GetEssential(functions[i], "Magnitude")
            if end_time <= start_time:
                raise ValueError("The interval time should be larger than zero")
            self.function[i - 1].set_value(start_time, end_time, period, phase, magnitude)

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
