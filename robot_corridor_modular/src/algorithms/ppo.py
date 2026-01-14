from __future__ import annotations # Permet d'utiliser des types pas encore définis (string forward ref)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Ces imports ne se font qu'au moment du check statique (IDE, mypy)
    # Pas au runtime, donc pas de crash circulaire.
    from src.environment.wrapper import CorridorEnv
    from src.core.types import ModelInput

class PPOAgent:
    # On utilise les guillemets ou le __future__ import
    def __init__(self, env: 'CorridorEnv'): 
        self.env = env