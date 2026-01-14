from typing import Protocol, Tuple, Any
import numpy as np
import numpy.typing as npt
from src.core.types import Vec3

class EnvironmentProtocol(Protocol):
    def step(self, action: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        # ...
        pass
    
    def reset(self) -> Tuple[npt.NDArray[np.float64], dict[str, Any]]:
        # ...
        pass
        
    def close(self) -> None:
        # ...
        pass

class RewardStrategyProtocol(Protocol):
    def compute(self, pos: Vec3, velocity: Vec3, action: npt.NDArray[np.float64], step_count: int, is_stuck: bool, is_backward: bool) -> float:
        # ...
        pass