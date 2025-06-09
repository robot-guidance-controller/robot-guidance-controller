import numpy as np
from numpy.typing import NDArray


class Intrinsics:
    def __init__(self, width: float, height: float, fx: float, fy: float, px: float, py: float, orthographic: bool = False) -> None:
        self.fx, self.fy = fx / width, fy / height
        self.px, self.py = px / width, py / height
        self.orthographic = orthographic

    def camera_to_screen(self, XYZ: NDArray, width: int, height: int) -> NDArray:
        if self.orthographic:
            x = XYZ[..., 0] * self.fx + self.px
            y = XYZ[..., 1] * self.fy + self.py
        else:
            x = XYZ[..., 0] / (XYZ[..., 2] + 1e-6) * self.fx + self.px
            y = XYZ[..., 1] / (XYZ[..., 2] + 1e-6) * self.fy + self.py

        z = XYZ[..., 2]

        return np.stack([width * x, height * y, z], axis=-1)

    def screen_to_camera(self, xyz: NDArray, width: int, height: int) -> NDArray:
        if self.orthographic:
            X = (xyz[..., 0] / width - self.px) / self.fx
            Y = (xyz[..., 1] / height - self.py) / self.fy
        else:
            X = ((xyz[..., 0] / width - self.px) / self.fx) * xyz[..., 2]
            Y = ((xyz[..., 1] / height - self.py) / self.fy) * xyz[..., 2]

        Z = xyz[..., 2]

        return np.stack([X, Y, Z], axis=-1)


class Camera:
    def __init__(self, intrinsics: Intrinsics, position: NDArray, rotation_matrix: NDArray) -> None:
        self.intrinsics = intrinsics
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    def world_to_camera(self, XYZ: NDArray) -> NDArray:
        XYZ = XYZ - self.position
        XYZ = XYZ @ self.inverse_rotation_matrix.T
        return XYZ

    def camera_to_screen(self, XYZ: NDArray, width: int, height: int) -> NDArray:
        return self.intrinsics.camera_to_screen(XYZ, width, height)

    def world_to_screen(self, XYZ: NDArray, width: int, height: int) -> NDArray:
        return self.camera_to_screen(self.world_to_camera(XYZ), width, height)

    def screen_to_camera(self, xyz: NDArray, width: int, height: int) -> NDArray:
        return self.intrinsics.screen_to_camera(xyz, width, height)

    def camera_to_world(self, XYZ: NDArray) -> NDArray:
        XYZ = XYZ @ self.rotation_matrix.T
        XYZ += self.position
        return XYZ

    def screen_to_world(self, xyz: NDArray, width: int, height: int) -> NDArray:
        return self.camera_to_world(self.screen_to_camera(xyz, width, height))

    def get_clip_mask(self, xyz: NDArray, width: int, height: int, depth: NDArray = None) -> NDArray:
        is_valid = (xyz[..., 0] >= 0) & (xyz[..., 0] <= width - 1)
        is_valid &= (xyz[..., 1] >= 0) & (xyz[..., 1] <= height - 1)
        is_valid &= (xyz[..., 2] > 0)

        if depth is not None:
            x_index = np.clip(
                xyz[..., 0] * depth.shape[1] / width, 0, depth.shape[1] - 1).astype(np.int64)
            y_index = np.clip(
                xyz[..., 1] * depth.shape[0] / height, 0, depth.shape[0] - 1).astype(np.int64)
            is_valid &= (xyz[..., 2] * 0.9 < depth[y_index, x_index])

        return is_valid

    def clip_xyz(self, xyz: NDArray, width: int, height: int, depth: NDArray = None) -> NDArray:
        is_valid = self.get_clip_mask(xyz, width, height, depth)
        return xyz[is_valid]

    def clip_XYZ(self, XYZ: NDArray, width: int, height: int, depth: NDArray = None) -> NDArray:
        xyz = self.world_to_screen(XYZ, width, height)
        is_valid = self.get_clip_mask(xyz, width, height, depth)
        return XYZ[is_valid]

    def forward(self) -> NDArray:
        return np.array([0.0, 0.0, 1.0]) @ self.rotation_matrix.T

    def calibrate(self, transform: NDArray):
        self.position = transform[0:3, 0:3] @ self.position + transform[0:3, 3]
        self.rotation_matrix = transform[0:3, 0:3] @ self.rotation_matrix
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)
