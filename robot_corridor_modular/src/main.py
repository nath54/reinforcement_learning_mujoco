import argparse
import multiprocessing as mp

from src.core.config_loader import load_config
from src.main_train import train
from src.main_play import play
from src.main_interactive import interactive

def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Corridor RL Training")
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play with trained model')
    parser.add_argument('--interactive', action='store_true', help='Interactive keyboard control')
    parser.add_argument('--render_mode', action='store_true', help='Replay saved controls')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for play mode')
    parser.add_argument('--live_vision', action='store_true', help='Show live vision window')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    if args.train:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        train(cfg)
    elif args.play:
        model_path = args.model_path if args.model_path else cfg.training.model_path
        play(cfg, model_path, args.live_vision)
    elif args.interactive:
        interactive(cfg, args.render_mode)
    else:
        print("Please specify mode: --train, --play, or --interactive")

if __name__ == "__main__":
    main()