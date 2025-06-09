from rgbd_stream import RGBDStream
import numpy as np
from numpy.typing import NDArray
import cv2


class CameraFeed:
    def __init__(self, window_title: str, stream: RGBDStream, calibration_matrix: NDArray = np.eye(4)) -> None:
        self.window_title = window_title
        self.stream = stream
        self.calibration_matrix = calibration_matrix

        if not self.stream.is_running():
            self.stream.start()

        self._get_new_frame()

    def _get_new_frame(self) -> None:
        self.stream.wait_for_frames()
        self.frame = self.stream.get_frame()
        self.frame.camera.calibrate(self.calibration_matrix)
        self.rgb = np.copy(self.frame.rgb)

    def draw_world_point(self, point: NDArray, radius: int, color: NDArray) -> None:
        width, height = self.rgb.shape[1], self.rgb.shape[0]
        x, y, z = self.frame.camera.world_to_screen(point, width, height)

        if z < 0:
            return

        cv2.circle(self.rgb, (int(x), int(y)),
                   radius=radius, color=color, thickness=-1)

    def draw_world_arrow(self, point1: NDArray, point2: NDArray, thickness: int, color: NDArray) -> None:
        width, height = self.rgb.shape[1], self.rgb.shape[0]
        x1, y1, z1 = self.frame.camera.world_to_screen(point1, width, height)
        x2, y2, z2 = self.frame.camera.world_to_screen(point2, width, height)

        if z1 < 0 or z2 < 0:
            return

        cv2.arrowedLine(self.rgb, (int(x1), int(y1)),
                        (int(x2), int(y2)), color=color, thickness=thickness)

    def update_window(self) -> None:
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_title, bgr)
        cv2.waitKey(1)

        self._get_new_frame()

    def save_frame(self, directory: str, name: str) -> None:
        cv2.imwrite(f'{directory}/{name}_rgb.bmp',
                    (cv2.cvtColor(self.frame.rgb, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8))
        cv2.imwrite(f'{directory}/{name}_d.bmp',
                    (np.clip(self.frame.depth, 0.0, 16.0) * 1000.0).astype(np.uint16))
