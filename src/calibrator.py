from __future__ import annotations
from rgbd_stream import RGBDStream, RGBDFrame
from robot import Robot
from numpy.typing import NDArray
import numpy as np
from typing import Tuple
import cv2


class Calibrator:
    def __init__(self, stream: RGBDStream, robot: Robot, marker_color: NDArray, N: int = 1000) -> None:
        self.robot = robot
        self.stream = stream
        self.marker_color = marker_color
        self.N = N
        self.world_space_points = []
        self.robot_space_points = []
        self.calibration_matrix = None

        if not self.stream.is_running():
            self.stream.start()

    def _find_marker_position(self, frame: RGBDFrame) -> Tuple[NDArray | None, NDArray]:
        difference = frame.resized_rgb - self.marker_color
        mask = np.sum(difference * difference, axis=-1) < 0.03
        indices_y, indices_x = np.where(mask)

        if len(indices_x) == 0:
            return None, mask

        marker_points = frame.camera.screen_to_world(
            frame.xyz[indices_y, indices_x],
            width=frame.xyz.shape[1],
            height=frame.xyz.shape[0])

        return np.mean(marker_points, axis=0), mask

    def _get_robot_position(self) -> NDArray:
        return self.robot.get_pose(Robot.TRANSLATION)

    def _kabsch_algorithm(self, P: NDArray, Q: NDArray) -> Tuple[NDArray, NDArray]:
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_Q - R @ centroid_P

        return R, t

    def is_calibrating(self) -> bool:
        return len(self.world_space_points) < self.N

    def calibrate(self, display_frame: bool = True) -> None:
        self.stream.wait_for_frames()
        frame = self.stream.get_frame()

        marker_position, mask = self._find_marker_position(frame)

        if display_frame:
            mask = cv2.resize(
                mask.astype(np.uint8), (frame.rgb.shape[1], frame.rgb.shape[0]))[:, :, None]
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR) * (1 - mask)
            cv2.imshow('Camera Feed', bgr)
            cv2.waitKey(1)

        if marker_position is None:
            return

        robot_position = self._get_robot_position()

        self.world_space_points.append(marker_position)
        self.robot_space_points.append(robot_position)

        print(f"Calibrating: {100.0 * len(self.world_space_points) / self.N}%")

    def compute_calibration_matrix(self) -> NDArray:
        rotation, translation = self._kabsch_algorithm(
            np.array(self.world_space_points), np.array(self.robot_space_points))

        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation
        matrix[0:3, 3] = translation

        return matrix
