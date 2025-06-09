from timer import Timer
import threading


class AppLoop:
    def __init__(self) -> None:
        self.timer = Timer()
        self.stop_event = threading.Event()

    def startup(self) -> None:
        pass

    def update(self, t: float, dt: float) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_event.set()

    def is_running(self) -> bool:
        return not self.stop_event.is_set()

    def run(self) -> None:
        self.startup()
        self.timer.reset()

        try:
            while self.is_running():
                t = self.timer.t()
                dt = self.timer.dt()
                self.update(t, dt)
        finally:
            self.shutdown()

    def run_threaded(self) -> None:
        thread = threading.Thread(target=self.run)
        thread.start()
