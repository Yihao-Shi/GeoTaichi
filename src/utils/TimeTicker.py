from time import time

class Timer(object):
    def __init__(self, description):
        self.description = description
        self.stack = []
        self.index = dict()
        self.flags = []
        self.levels = []
        self.timings = []

    def __enter__(self):
        self.stack.append(self.description)
        if self.description not in self.index:
            self.index[self.description] = len(self.flags)
            self.flags.append(self.description)
            self.levels.append(len(self.stack))
            self.timings.append(0.0)
        id = self.index[self.description]
        self.timings[id] -= time()

    def __exit__(self, type, value, traceback):
        self.stack.pop()
        id = self.index[self.description]
        self.timings[id] += time()

class FpsCounter:
    nanosecs_per_sec = 1e9
    def __init__(self, burn_in_secs: float = 5.0, print_period_secs: float = 5.0):
        self.t0 = time.monotonic_ns()
        self.base_frame = -1
        self.burn_in_time_ns = burn_in_secs * self.nanosecs_per_sec
        self.print_period_nanosecs = int(print_period_secs * self.nanosecs_per_sec)
        self.avg_fps = -1.0
        self.elapsed_secs = -1.0

    def count_fps(self, frame: int):
        t1 = time.monotonic_ns()
        if self.base_frame == -1 and t1 - self.t0 > self.burn_in_time_ns:
            self.base_frame = frame
            self.t0 = t1
            self.t_last_tick = t1

        if self.base_frame > 0 and t1 - self.t_last_tick > self.print_period_nanosecs:
            self.t_last_tick = t1
            self.avg_fps = self.nanosecs_per_sec * (frame - self.base_frame) / (t1 - self.t0)
            self.elapsed_secs = (t1 - self.t0) / self.nanosecs_per_sec

            print(f"Avg FPS: {self.avg_fps}     (after {self.elapsed_secs}s)")

        return (self.avg_fps, self.elapsed_secs)