# Projet PPO Robot Corridor

## Organization

```
my_robot_rl/
├── config/                          # TOUTES les configurations YAML
│   ├── main.yaml                    # Point d'entrée (pointe vers les sous-configs)
│   ├── agents/                      # Choix du modèle (MLP, Transformer, etc.)
│   │   ├── policy_mlp_small.yaml
│   │   ├── policy_transformer_large.yaml
│   │   └── actor_cnn_default.yaml
│   ├── environments/                # Configs de la map et physique
│   │   ├── corridor_standard.yaml
│   │   └── corridor_hard.yaml
│   └── rewards/                     # Configs des fonctions de récompense
│       ├── standard_reward.yaml
│       └── velocity_focused.yaml
│
├── src/
│   ├── __init__.py
│   ├── core/                        # NIVEAU 0 : Pas de dépendances externes
│   │   ├── __init__.py
│   │   ├── types.py                 # Dataclasses, Enums (Vec3, Rect2d)
│   │   ├── interfaces.py            # Protocoles/ABC pour éviter les refs circulaires
│   │   └── config_loader.py         # Chargeur YAML intelligent
│   │
│   ├── simulation/                  # NIVEAU 1 : MuJoCo & Physique
│   │   ├── __init__.py
│   │   ├── generator.py             # Génération procédurale (Corridor)
│   │   ├── robot.py                 # Gestion XML du robot
│   │   ├── physics.py               # Moteur physique (forces, drag)
│   │   └── sensors.py               # Vision et Collisions
│   │
│   ├── environment/                 # NIVEAU 2 : Gym Wrapper & Logique
│   │   ├── __init__.py
│   │   ├── wrapper.py               # CorridorEnv (Gym)
│   │   └── reward_strategy.py       # Stratégies de récompenses interchangeables
│   │
│   ├── models/                      # NIVEAU 3 : Réseaux de Neurones (PyTorch)
│   │   ├── __init__.py
│   │   ├── factory.py               # Création dynamique des modèles
│   │   ├── mlp.py                   # Implémentation MLP
│   │   └── transformer.py           # Implémentation Transformer
│   │
│   ├── algorithms/                  # NIVEAU 4 : Algorithmes RL
│   │   ├── __init__.py
│   │   └── ppo.py                   # Agent PPO (utilise les modèles)
│   │
│   └── main.py                      # Point d'entrée
```
