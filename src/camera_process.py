import multiprocessing as mp
import numpy as np
import time
from typing import Callable, Dict, Any
from rgbd_stream import RGBDStream_iOS
from camera_feed import CameraFeed

class CameraProcess:
    def __init__(
        self,
        update_fn: Callable[[CameraFeed, Dict[str, Any]], None],
        init_pose: np.ndarray,
        calibration_path: str = 'calibration_matrix.npy',
    ):
        self.queue = mp.Queue()
        self.proc = mp.Process(
            target=self._worker,
            args=(self.queue, update_fn, init_pose, calibration_path)
        )

    def start(self):
        self.proc.start()

    def send(self, message: Dict[str, Any]):
        self.queue.put(message)

    def terminate(self):
        self.queue.put(None)
        self.proc.join()

    @staticmethod
    def _worker(queue, update_fn, init_pose, calibration_path):
        rgbd_stream = RGBDStream_iOS()
        calibration_matrix = np.load(calibration_path)
        camera_feed = CameraFeed("Camera Feed", rgbd_stream, calibration_matrix)

        # Init phase: render yellow dot until 'start' signal is received
        started = False
        while not started:
            camera_feed.draw_world_point(init_pose[:3], radius=10, color=(0xff, 0xff, 0))
            camera_feed.update_window()

            if not queue.empty():
                msg = queue.get()
                if msg is None:
                    return
                if msg.get("start", False):
                    started = True
            else:
                time.sleep(0.01)

        # Main loop: draw user-defined elements
        while True:
            if not queue.empty():
                msg = queue.get()
                if msg is None:
                    break
                update_fn(camera_feed, msg)
                camera_feed.update_window()
            else:
                time.sleep(0.01)