import taichi as ti

from contextlib import contextmanager
from time import perf_counter
import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import Threshold


def time_tolerance(timestep):
    return max(Threshold, 1.0e-9 * abs(float(timestep)))


def has_remaining_time(current_time, target_time, timestep):
    return float(target_time) - float(current_time) > time_tolerance(timestep)


def advance_time(sims, timestep):
    """Kahan summation prevents roundoff from creating a tiny physical step."""
    last_time, correction = getattr(sims, "_time_sum", (sims.current_time, 0.0))
    if last_time != sims.current_time:  # A restart or caller reset the clock.
        correction = 0.0
    increment = float(timestep) - correction
    updated = sims.current_time + increment
    sims._time_sum = (updated, (updated - sims.current_time) - increment)
    sims.current_time = updated


class TimerRecord(object):
    def __init__(self, name):
        self.name = str(name)
        self.total = 0.0
        self.current = 0.0
        self.num = 0
        self.start = 0.0

    def begin(self):
        self.start = perf_counter()

    def end(self):
        end = perf_counter()
        cur_time = end - self.start
        self.total += cur_time
        self.current = cur_time
        self.num += 1

    def profile(self):
        return self.current, self.total / self.num


class Timer(object):
    def __init__(self):
        self.records = {}

    def begin(self, name):
        if name in self.records.keys():
            self.records[name].begin()
        else:
            self.records.update({name: TimerRecord(name)})
            self.records[name].begin()

    def end(self, name):
        if GlobalVariable.USEGPU:
            ti.sync()
        if name not in self.records:
            self.begin(name)
        self.records[name].end()

    @contextmanager
    def section(self, name):
        self.begin(name)
        try:
            yield
        except BaseException as exception:
            try:
                self.end(name)
            except BaseException as end_error:
                raise exception from end_error
            raise
        else:
            self.end(name)

    def profile0(self):
        msg = "#     Time record accmulated(execute num): "
        total_time = 0.0
        for name, rec in self.records.items():
            msg += f"{name}: {rec.total:.3f}({rec.num}), "
            total_time += rec.total
        msg += f"total: {total_time:.3f} s"
        print(msg)

    def profile1(self):
        msg = "#     Time record cur(avg): "
        total_time = 0.0
        for name, rec in self.records.items():
            info = rec.profile()
            msg += f"{name}: {info[0]:.3f}({info[1]:.3f}), "
            total_time += rec.total
        msg += f"total: {total_time:.3f} s"
        print(msg)
