import argparse
from src.core.config_loader import load_config
from src.environment.wrapper import CorridorEnv
from src.algorithms.ppo import PPOAgent
from src.simulation.controls import Controls
from src.simulation.sensors import Camera

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/main.yaml')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = CorridorEnv(cfg)
    
    # Calculate dim
    obs_dim = env.observation_space.shape[0]
    act_dim = 4 if cfg.robot.control_mode == "discrete_direction" else 2
    
    agent = PPOAgent(cfg.model, obs_dim, act_dim, cfg.training.learning_rate)
    
    if args.train:
        print(f"Starting training on {cfg.simulation.corridor_length}m corridor...")
        obs, _ = env.reset()
        for i in range(cfg.simulation.max_steps):
            action, _ = agent.select_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
    else:
        # Play mode logic with Controls
        print("Play mode (Visual only)")
        # ... viewer loop ...

if __name__ == "__main__":
    main()