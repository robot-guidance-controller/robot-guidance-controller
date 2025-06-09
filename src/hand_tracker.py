from rgbd_stream import RGBDFrame
import mediapipe as mp
import numpy as np
import time
from numpy.typing import NDArray


class HandTracker:
    def __init__(self):
        self.hand_positions = []
        self.hand_times = []
        self.hand_velocities = []
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def update(self, frame: RGBDFrame) -> None:
        results = self.hand_detector.process(
            (frame.rgb * 255.0).astype(np.uint8))
        if results.multi_hand_landmarks:
            landmark = results.multi_hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
            x, y = landmark.x, landmark.y
            if x >= 0.0 and y >= 0.0 and x < 1.0 and y < 1.0:
                width, height = frame.resized_depth.shape[1], frame.resized_depth.shape[0]
                z = frame.resized_depth[int(y * height), int(x * width)]
                xyz = np.array([x * width, y * height, z])
                XYZ = frame.camera.screen_to_world(xyz, width, height)

                self.hand_positions.append(XYZ)
                self.hand_times.append(time.time())
                if len(self.hand_positions) > 1:
                    dt = self.hand_times[-1] - self.hand_times[-2]
                    velocity = (
                        self.hand_positions[-1] - self.hand_positions[-2]) / dt
                    self.hand_velocities.append(velocity)
                else:
                    self.hand_velocities.append(np.zeros(3))

    def get_position(self) -> NDArray:
        if len(self.hand_positions) > 0:
            return self.hand_positions[-1]
        return np.zeros(3)

    def get_velocity(self) -> NDArray:
        if len(self.hand_velocities) > 0:
            return self.hand_velocities[-1]
        return np.zeros(3)
