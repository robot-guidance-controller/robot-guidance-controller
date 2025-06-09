import time


class Timer:
    def __init__(self) -> None:
        self.reset()

    def dt(self) -> float:
        t = self.t()
        dt = t - self.last_t
        self.last_t = t
        return dt

    def t(self) -> float:
        return time.time() - self.start_time

    def reset(self) -> None:
        self.start_time = time.time()
        self.last_t = 0.0
