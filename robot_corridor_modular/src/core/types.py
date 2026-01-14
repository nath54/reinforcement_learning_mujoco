from dataclasses import dataclass
import numpy as np

# Tout le monde a besoin de Vec3, donc on le met ici.
@dataclass
class Vec3:
    x: float
    y: float
    z: float

# On définit les inputs/outputs du réseau ici pour que l'Env et l'Agent parlent le même langage
@dataclass
class ModelInput:
    vision: np.ndarray
    state_vector: np.ndarray