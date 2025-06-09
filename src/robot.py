from typing import List, Optional, Union
import rtde_receive
import rtde_control
import numpy as np
from timer import Timer
import time

class Robot:
    TRANSLATION_ROTATION = (0, 1, 2, 3, 4, 5)
    X, Y, Z, THETA_X, THETA_Y, THETA_Z = TRANSLATION_ROTATION
    TRANSLATION = (X, Y, Z)
    ROTATION = (THETA_X, THETA_Y, THETA_Z)
    TRANSLATION_ROTATION_SEPARATED = (TRANSLATION, ROTATION)

    @staticmethod
    def _extract_axes(translation_rotation: List[float], axes: Union[int, List[int]]):
        if isinstance(axes, int):
            return translation_rotation[axes]
        return np.asarray([translation_rotation[ax] for ax in axes])

    def get_axes(self, translation_rotation: List[float], axes: Optional[Union[int, List[int], List[List[int]]]] = None):
        if axes is None:
            axes = self.default_axes

        if isinstance(axes, int):
            return translation_rotation[axes]
        elif isinstance(axes, (tuple, list)):
            if all(isinstance(subset, (tuple, list)) for subset in axes):
                return tuple(Robot._extract_axes(translation_rotation, subset) for subset in axes)
        return Robot._extract_axes(translation_rotation, axes)

    @staticmethod
    def _update_axes(translation_rotation: List[float], input: List[float], axes: List[int]):
        for i, ax in enumerate(axes):
            translation_rotation[ax] = input[i]

    def set_axes(self, translation_rotation: List[float], input: Union[float, List[float]], axes: Optional[Union[int, List[int]]] = None, reset_unspecified: bool = False):
        if axes is None:
            axes = self.default_axes

        if isinstance(axes, int):
            axes = [axes]
            input = [input]

        if isinstance(axes, (tuple, list)):
            if all(isinstance(subset, (tuple, list)) for subset in axes):
                for subset_input, subset_axes in zip(input, axes):
                    Robot._update_axes(translation_rotation, subset_input, subset_axes)
                axes = set(ax for subset_axes in axes for ax in subset_axes)
            else:
                Robot._update_axes(translation_rotation, input, axes)

        if reset_unspecified:
            for i in Robot.TRANSLATION_ROTATION:
                if i not in axes:
                    translation_rotation[i] = 0.0

    def zeroed_wrench(self, axes: Optional[Union[int, List[int], List[List[int]]]] = None):
        if axes is None:
            axes = self.default_axes
        if isinstance(axes, int):
            return 0
        return np.zeros(len(axes))

    def __init__(
        self,
        ip: str,
        default_axes: Optional[Union[int, List[int], List[List[int]]]] = None,
        init_pose: Optional[List[float]] = None,
        velocity_input_ewma_tau: Optional[float] = None,
        force_ewma_tau: Optional[float] = None,
        translational_force_deadband: Optional[float] = None,
        rotational_torque_deadband: Optional[float] = None,
    ):
        self.receive = rtde_receive.RTDEReceiveInterface(ip)
        self.control = rtde_control.RTDEControlInterface(ip)

        self.default_axes = Robot.TRANSLATION_ROTATION

        self._pose_input = self.zeroed_wrench()

        self._velocity_input = self.zeroed_wrench()
        self._velocity_input_ewma_tau = velocity_input_ewma_tau
        self._prev_velocity_input = self.zeroed_wrench()
        self._velocity_timer = Timer()
        
        self._force_ewma_tau = force_ewma_tau
        self._prev_force = self.zeroed_wrench()
        self._force_timer = Timer()

        self._translational_force_deadband = translational_force_deadband
        self._rotational_torque_deadband = rotational_torque_deadband

        if init_pose is not None:
            self.set_pose(init_pose)
            init_pose_delay_timer = Timer()
            while init_pose_delay_timer.t() < 1:
                period_start = self.control.initPeriod()
                self.set_velocity(self.zeroed_wrench(), reset_unspecified=True)
                self.control.waitPeriod(period_start)

        self.INIT_POSE = self.get_pose()
        self._pose_input = self.INIT_POSE

        if default_axes is not None:
            self.default_axes = default_axes

    def get_pose(self, axes: Optional[Union[int, List[int], List[List[int]]]] = None):
        return self.get_axes(self.receive.getActualTCPPose(), axes)

    def set_pose(
        self,
        input: Union[float, List[float]],
        axes: Optional[Union[int, List[int]]] = None,
        reset_unspecified: bool = False,
        speed: float = 0.25,
        acceleration: float = 1.2,
        asynchronous: bool = False,
    ):
        self.set_axes(self._pose_input, input, axes, reset_unspecified)
        self.control.moveL(self._pose_input, speed, acceleration, asynchronous)

    def get_velocity(self, axes: Optional[Union[int, List[int], List[List[int]]]] = None):
        return self.get_axes(self.receive.getActualTCPSpeed(), axes)

    def set_velocity(
        self,
        input: Union[float, List[float]],
        axes: Optional[Union[int, List[int]]] = None,
        reset_unspecified: bool = False,
        acceleration: float = 0.25,
        time: float = 0.0,
    ):
        self.set_axes(self._velocity_input, input, axes, reset_unspecified)

        if self._velocity_input_ewma_tau is not None:
            alpha = 1 - np.exp(-self._velocity_timer.dt() / self._velocity_input_ewma_tau)
            self._velocity_input = (
                alpha * self._velocity_input + (1 - alpha) * self._prev_velocity_input
            )
            self._prev_velocity_input = self._velocity_input

        self.control.speedL(self._velocity_input, acceleration, time)

    def get_force(self, axes: Optional[Union[int, List[int], List[List[int]]]] = None):
        force = np.array(self.receive.getActualTCPForce())

        for deadband, axes_to_deadband in zip(
            (self._translational_force_deadband, self._rotational_torque_deadband),
            Robot.TRANSLATION_ROTATION_SEPARATED,
        ):
            magnitude = np.linalg.norm(self.get_axes(force, axes_to_deadband))

            if deadband is not None and magnitude < deadband:
                new_magnitude = max(0, 2 * magnitude - deadband)
                new_force = new_magnitude * force / magnitude
                self.set_axes(force, new_force, axes_to_deadband)

        if self._force_ewma_tau is not None:
            alpha = 1 - np.exp(-self._force_timer.dt() / self._force_ewma_tau)
            force = alpha * force + (1 - alpha) * self._prev_force
            self._prev_force = force

        return self.get_axes(force, axes)

    def __enter__(self):
        self.control.zeroFtSensor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_velocity(self.zeroed_wrench(Robot.TRANSLATION_ROTATION), Robot.TRANSLATION_ROTATION, reset_unspecified=True)