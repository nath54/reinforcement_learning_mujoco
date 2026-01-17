from typing import Protocol, Any

import numpy as np
from numpy.typing import NDArray

from src.core.types import Vec3


#
class EnvironmentProtocol(Protocol):

    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        # ...
        pass

    def reset(self) -> tuple[NDArray[np.float64], dict[str, Any]]:
        # ...
        pass

    def close(self) -> None:
        # ...
        pass


#
class RewardStrategyProtocol(Protocol):

    def compute(self, pos: Vec3, velocity: Vec3, action: NDArray[np.float64], step_count: int, is_stuck: bool, is_backward: bool) -> float:
        # ...
        pass