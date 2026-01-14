from typing import Protocol, Any, Tuple
import numpy as np

# L'agent n'a pas besoin de connaître "CorridorEnv", juste qu'il y a un truc qui step()
class EnvironmentProtocol(Protocol):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # ...
        pass
    def reset(self) -> Tuple[np.ndarray, dict]:
        # ...
        pass