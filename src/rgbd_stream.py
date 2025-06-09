import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from record3d import Record3DStream
from threading import Event
import cv2
from camera import Camera, Intrinsics


def quaternion_to_matrix(quaternion: NDArray) -> NDArray:
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return np.array([
        [1 - 2 * (y * y + z * z),
         2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w),
         1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w),
         2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)]
    ])


class RGBDFrame:
    def __init__(self, rgb: NDArray, depth: NDArray, confidence: NDArray, camera: Camera) -> None:
        self.rgb = rgb
        self.depth = depth
        self.confidence = confidence
        self.camera = camera

        self.resized_rgb = cv2.resize(rgb, np.flip(depth.shape))
        self.resized_depth = cv2.resize(depth, np.flip(rgb.shape[0:2]))

        x, y = np.meshgrid(
            np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        self.xyz = np.stack([x + 0.5, y + 0.5, depth], axis=-1)

    def compute_XYZ(self) -> None:
        self.XYZ = self.camera.screen_to_world(
            self.xyz, self.depth.shape[1], self.depth.shape[0])

    def get_normals(self) -> NDArray:
        down_shift_XYZ = np.roll(self.XYZ, 1, axis=0)
        right_shift_XYZ = np.roll(self.XYZ, 1, axis=1)

        normals = np.cross(down_shift_XYZ - self.XYZ,
                           right_shift_XYZ - self.XYZ, axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        return normals

    def get_filtered_world_points(self, min_confidence=0) -> Tuple[NDArray, NDArray]:
        valid_depth = self.confidence >= min_confidence

        XYZ = self.XYZ[valid_depth]
        rgb = self.resized_rgb[valid_depth]

        return XYZ, rgb.astype(np.float64)

    def get_carved_points_mask(self, XYZ: NDArray) -> NDArray:
        xyz = self.camera.world_to_screen(
            XYZ, self.depth.shape[1], self.depth.shape[0])

        is_valid = (xyz[..., 0] < 0.0) | (xyz[..., 0] >= self.depth.shape[1])
        is_valid |= (xyz[..., 1] < 0.0) | (xyz[..., 1] >= self.depth.shape[0])
        is_valid |= (xyz[..., 2] < 0.0)

        x_index = np.clip(xyz[:, 0].astype(np.int32),
                          0, self.depth.shape[1] - 1)
        y_index = np.clip(xyz[:, 1].astype(np.int32),
                          0, self.depth.shape[0] - 1)

        error_margin = 2.5 - self.confidence[y_index, x_index] / 2.0
        is_valid |= xyz[..., 2] * error_margin >= self.depth[y_index, x_index]

        return is_valid

    def carve_points(self, XYZ: NDArray) -> NDArray:
        return XYZ[self.get_carved_points_mask(XYZ)]


class RGBDStream:
    '''Abstract base class for different RGBD stream backends'''

    def __init__(self) -> None:
        raise NotImplementedError("Use an RGBDStream subclass")

    def start(self) -> None:
        raise NotImplementedError()

    def is_running(self) -> bool:
        raise NotImplementedError()

    def stop(self) -> None:
        raise NotImplementedError()

    def wait_for_frames(self) -> None:
        raise NotImplementedError()

    def get_frame(self) -> RGBDFrame:
        raise NotImplementedError()


class RGBDStream_iOS(RGBDStream):
    def __init__(self, device_index=0) -> None:
        devices = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devices)))
        for device in devices:
            print('\tID: {}\n\tUDID: {}\n'.format(
                device.product_id, device.udid))

        if len(devices) <= device_index:
            raise RuntimeError(
                f'Cannot connect to device #{device_index}, try different index.')

        device = devices[device_index]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.stop
        self.session.connect(device)

        self.event = Event()
        self.streaming = False

    def on_new_frame(self) -> None:
        self.event.set()

    def start(self) -> None:
        self.streaming = True

    def is_running(self) -> bool:
        return self.streaming

    def stop(self) -> None:
        self.streaming = False

    def wait_for_frames(self) -> None:
        self.event.wait()
        self.event.clear()

    def _get_rgb_frame(self) -> NDArray:
        rgb = self.session.get_rgb_frame()
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rgb.astype(np.float32) / 255.0

    def _get_depth_frame(self) -> NDArray:
        depth = self.session.get_depth_frame()
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return depth.astype(np.float32)

    def _get_confidence_frame(self) -> NDArray:
        confidence = self.session.get_confidence_frame()
        confidence = cv2.rotate(confidence, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return confidence

    def get_camera(self) -> Camera:
        intrinsics = self.session.get_intrinsic_mat()
        extrinsics = self.session.get_camera_pose()

        rgb_shape = self.session.get_rgb_frame().shape
        intrinsics = Intrinsics(
            width=rgb_shape[0], height=rgb_shape[1],
            fx=intrinsics.fy, fy=intrinsics.fx,
            px=intrinsics.ty, py=rgb_shape[1] - 1 - intrinsics.tx)

        position = np.array([extrinsics.tx, -extrinsics.ty, -extrinsics.tz])
        quaternion = np.array(
            [extrinsics.qx, -extrinsics.qy, -extrinsics.qz, extrinsics.qw])

        camera = Camera(intrinsics, position, quaternion_to_matrix(quaternion))
        camera.inverse_rotation_matrix[[0, 1]] = \
            camera.inverse_rotation_matrix[[1, 0]]
        camera.inverse_rotation_matrix[1] *= -1.0
        camera.rotation_matrix = np.linalg.inv(camera.inverse_rotation_matrix)

        return camera

    def get_frame(self) -> RGBDFrame:
        rgb = self._get_rgb_frame()
        depth = self._get_depth_frame()
        confidence = self._get_confidence_frame()
        camera = self.get_camera()
        return RGBDFrame(rgb, depth, confidence, camera)
