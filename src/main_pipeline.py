"""
Training Pipeline Runner

Runs a curriculum of training stages, automatically loading weights from
previous stages and supporting early stopping per stage.

Usage:
    python -m src.main_pipeline --pipeline configs_pipelines/curriculum_v1/pipeline.yaml
"""

import argparse
import os
import sys
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import load_config
from src.main_train import train


def load_pipeline_config(pipeline_path: str) -> Dict[str, Any]:
    """Load pipeline configuration YAML."""
    with open(pipeline_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict[str, Any], stage_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge stage overrides into base config."""
    import copy
    result = copy.deepcopy(base_config)

    for key, value in stage_overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def run_pipeline(pipeline_path: str, output_dir: Optional[str] = None) -> None:
    """
    Execute a training pipeline.

    Args:
        pipeline_path: Path to pipeline.yaml
        output_dir: Override output directory (default: auto-generated)
    """
    pipeline_dir = Path(pipeline_path).parent
    pipeline_config = load_pipeline_config(pipeline_path)

    pipeline_name = pipeline_config.get('name', 'unnamed_pipeline')
    stages = pipeline_config.get('stages', [])
    shared_config = pipeline_config.get('shared', {})

    if not stages:
        print("Error: No stages defined in pipeline")
        return

    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if output_dir is None:
        output_dir = f"trainings_pipeline/{timestamp}_{pipeline_name.replace(' ', '_')}"

    os.makedirs(output_dir, exist_ok=True)

    # Save pipeline config for reference
    shutil.copy(pipeline_path, os.path.join(output_dir, "pipeline.yaml"))

    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE: {pipeline_name}")
    print(f"Output: {output_dir}")
    print(f"Stages: {len(stages)}")
    print(f"{'='*60}\n")

    current_weights_path: Optional[str] = None

    for stage_idx, stage in enumerate(stages):
        stage_name = stage.get('name', f'Stage {stage_idx + 1}')
        stage_config_file = stage.get('config')
        max_episodes = stage.get('max_episodes', 10000)
        early_stop_successes = stage.get('early_stop_successes', 0)

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage_name}")
        print(f"Config: {stage_config_file}")
        print(f"Max episodes: {max_episodes}")
        if early_stop_successes > 0:
            print(f"Early stop after: {early_stop_successes} consecutive successes")
        print(f"{'='*60}\n")

        # Load stage config
        stage_config_path = pipeline_dir / stage_config_file
        if not stage_config_path.exists():
            print(f"Error: Stage config not found: {stage_config_path}")
            return

        # Load and merge configs
        with open(stage_config_path, 'r') as f:
            stage_yaml = yaml.safe_load(f)

        # Apply shared config (pipeline-level overrides)
        if shared_config:
            stage_yaml = merge_configs(stage_yaml, {'model': shared_config})

        # Override training parameters
        if 'training' not in stage_yaml:
            stage_yaml['training'] = {}

        stage_yaml['training']['max_episodes'] = max_episodes

        if early_stop_successes > 0:
            stage_yaml['training']['early_stopping_enabled'] = True
            stage_yaml['training']['early_stopping_consecutive_successes'] = early_stop_successes

        # Load weights from previous stage
        if current_weights_path is not None:
            stage_yaml['training']['load_weights_from'] = current_weights_path
            print(f"Loading weights from: {current_weights_path}")

        # Set output path for this stage
        stage_output_dir = os.path.join(output_dir, f"stage_{stage_idx + 1:02d}_{stage_name.replace(' ', '_')}")
        os.makedirs(stage_output_dir, exist_ok=True)

        # Save merged config
        merged_config_path = os.path.join(stage_output_dir, "config.yaml")
        with open(merged_config_path, 'w') as f:
            yaml.dump(stage_yaml, f, default_flow_style=False)

        # Run training
        print(f"\nStarting training for stage {stage_idx + 1}...")

        try:
            config = load_config(merged_config_path)
            train(config, exp_dir_override=stage_output_dir)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print(f"Progress saved to: {stage_output_dir}")
            return
        except Exception as e:
            print(f"\nError in training stage {stage_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return

        # Update weights path for next stage
        best_model_path = os.path.join(stage_output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            current_weights_path = best_model_path
            print(f"\nStage {stage_idx + 1} complete. Best model: {best_model_path}")
        else:
            print(f"\nWarning: No best model found for stage {stage_idx + 1}")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"Final model: {current_weights_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run training pipeline')
    parser.add_argument('--pipeline', type=str, required=True,
                        help='Path to pipeline.yaml config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory')

    args = parser.parse_args()

    if not os.path.exists(args.pipeline):
        print(f"Error: Pipeline config not found: {args.pipeline}")
        return

    run_pipeline(args.pipeline, args.output)


if __name__ == "__main__":
    main()
