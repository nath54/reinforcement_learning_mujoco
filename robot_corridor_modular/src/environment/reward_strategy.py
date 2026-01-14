from abc import ABC, abstractmethod
from src.core.types import Vec3

class RewardStrategy(ABC):
    @abstractmethod
    def compute(self, pos: Vec3, velocity: Vec3, action: float) -> float:
        pass

class VelocityReward(RewardStrategy):
    def __init__(self, scale: float):
        self.scale = scale

    def compute(self, pos: Vec3, velocity: Vec3, action: float) -> float:
        return velocity.x * self.scale

class DistanceReward(RewardStrategy):
    def compute(self, pos: Vec3, velocity: Vec3, action: float) -> float:
        return pos.x * 0.1