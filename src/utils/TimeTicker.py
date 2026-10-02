import taichi as ti

from time import perf_counter
import src.utils.GlobalVariable as GlobalVariable


class TimerRecord(object):
    def __init__(self, name):
        self.name = str(name)
        self.total = 0.
        self.current = 0.
        self.num = 0
        self.start = 0.

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
        self.records[name].end()

    def profile0(self):
        msg = "#     Time record accmulated(execute num): "
        total_time = 0.
        for name, rec in self.records.items():
            msg += f"{name}: {rec.total:.3f}({rec.num}), "
            total_time += rec.total
        msg += f"total: {total_time:.3f} s"
        print(msg)

    def profile1(self):
        msg = "#     Time record cur(avg): "
        total_time = 0.
        for name, rec in self.records.items():
            info = rec.profile()
            msg += f"{name}: {info[0]:.3f}({info[1]:.3f}), "
            total_time += rec.total
        msg += f"total: {total_time:.3f} s"
        print(msg)
        
