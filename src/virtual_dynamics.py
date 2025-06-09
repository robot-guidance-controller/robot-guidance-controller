from numpy.typing import NDArray


class VirtualDynamics:
    def __init__(self, M: float | NDArray) -> None:
        self.M = M
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

    def apply_force(self, F: float | NDArray, dt: float) -> None:
        self.a = F / self.M
        self.x += self.v * dt + 0.5 * self.a * dt ** 2
        self.v += self.a * dt

    def get_position(self) -> float | NDArray:
        return self.x

    def get_velocity(self) -> float | NDArray:
        return self.v

    def get_acceleration(self) -> float | NDArray:
        return self.a


class SimpleVirtualDynamics(VirtualDynamics):
    def __init__(self, M: float | NDArray, B: float | NDArray, K: float | NDArray) -> None:
        super().__init__(M)
        self.B = B
        self.K = K

    def apply_force(self, F: float | NDArray, dt: float) -> None:
        F_net = F - self.B * self.v - self.K * self.x
        super().apply_force(F_net, dt)
