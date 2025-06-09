from typing import Any, Callable, Dict, Optional
import threading
import traceback

class StoppableThread(threading.Thread):
    def __init__(
        self, 
        stoppable_method: Callable[[threading.Event, Dict[Any, Any]], None], 
        stoppable_method_args: Optional[Any] = None, 
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.stoppable_method = stoppable_method
        self.stoppable_method_args = stoppable_method_args
        self.stop_event = threading.Event()

    def run(self):
        try:
            self.stoppable_method(self.stop_event, self.stoppable_method_args)
        except Exception as e:
            print(f'Error occurred in thread {self.name}: {e}')
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()