import argparse
import multiprocessing as mp

from src.core.config_loader import load_config
from src.main_train import train
from src.main_play import play
from src.main_interactive import interactive

#
def parse_args() -> argparse.Namespace:
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Robot Corridor RL Training")
    #
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play with trained model')
    parser.add_argument('--interactive', action='store_true', help='Interactive keyboard control')
    parser.add_argument('--pipeline', type=str, default=None, help='Run training pipeline from yaml')
    parser.add_argument('--render_mode', action='store_true', help='Replay saved controls')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for play mode')
    parser.add_argument('--live_vision', action='store_true', help='Show live vision window')
    #
    args: argparse.Namespace = parser.parse_args()
    #
    return args


#
def main() -> None:

    # Parse arguments
    args: argparse.Namespace = parse_args()

    # Launch the selected mode:

    # Run training pipeline of training runs
    if args.pipeline:
        #
        from src.main_pipeline import run_pipeline
        #
        run_pipeline(args.pipeline)

    # Run single training run
    elif args.train:
        #
        cfg = load_config(args.config)
        #
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        #
        train(cfg)

    # Agents play with trained model so we can see what they learned
    elif args.play:
        #
        cfg = load_config(args.config)
        #
        model_path = args.model_path if args.model_path else cfg.training.model_path
        #
        play(cfg, model_path, args.live_vision)

    # Interactive keyboard control
    elif args.interactive:
        #
        cfg = load_config(args.config)
        #
        interactive(cfg, args.render_mode)

    # Invalid mode
    else:
        #
        print("Please specify mode: --train, --play, --interactive, or --pipeline")

#
if __name__ == "__main__":
    #
    main()