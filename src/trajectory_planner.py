import pandas as pd
import numpy as np
from numpy.typing import NDArray

class TrajectoryPlanner:
    def __init__(self, reference_trajectory_csv_path: str, initial_position: NDArray):
        reference_trajectory_data = pd.read_csv(reference_trajectory_csv_path)

        self.positions = np.stack([reference_trajectory_data['p_x'], reference_trajectory_data['p_y'], reference_trajectory_data['p_z']], axis=1)
        self.positions += initial_position - self.positions[0]

        self.velocities = np.stack([reference_trajectory_data['v_x'], reference_trajectory_data['v_y'], reference_trajectory_data['v_z']], axis=1)

        self.going_forward = True

    def update_reference_trajectory(self, current_position) -> bool:
        if current_position[0] <= self.positions[-1, 0] if self.going_forward else current_position[0] >= self.positions[0, 0]:
            self.going_forward = not self.going_forward
            self.velocities *= -1
        return self.going_forward

    def get_closest_target(self, current_position: NDArray) -> tuple[NDArray, NDArray]:
        distances = np.linalg.norm(self.positions - current_position, axis=1)
        index = np.argmin(distances)
        return self.positions[index], self.velocities[index]