"""
Training Pipeline Runner

Runs a curriculum of training stages, automatically loading weights from
previous stages and supporting early stopping per stage.

Usage:
    python -m src.main_pipeline --pipeline configs_pipelines/curriculum_v1/pipeline.yaml

Or:
    python -m src.main --pipeline configs_pipelines/curriculum_v1/pipeline.yaml
"""

from typing import Any, Optional

import os
import sys
import copy
import yaml
import shutil
import argparse
import traceback

from pathlib import Path
from datetime import datetime

from main_train import train
from core.config_loader import load_config


# Load pipeline configuration YAML
def load_pipeline_config(pipeline_path: str) -> dict[str, Any]:
    """
    Load pipeline configuration YAML.
    """

    with open(pipeline_path, 'r') as f:
        return yaml.safe_load(f)


# Merge stage overrides into base config
def merge_configs(
    base_config: dict[str, Any],
    stage_overrides: dict[str, Any]
) -> dict[str, Any]:

    """
    Deep merge stage overrides into base config.
    """

    result: dict[str, Any] = copy.deepcopy(base_config)

    #
    key: str
    value: Any
    #
    for key, value in stage_overrides.items():
        #
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            #
            result[key] = merge_configs(result[key], value)
        #
        else:
            #
            result[key] = value

    return result


# Execute a training pipeline
def run_pipeline(pipeline_path: str, output_dir: Optional[str] = None) -> None:
    """
    Execute a training pipeline.

    Args:
        pipeline_path: Path to pipeline.yaml
        output_dir: Override output directory (default: auto-generated)
    """

    # Load pipeline config
    pipeline_dir: Path = Path(pipeline_path).parent
    pipeline_config: dict[str, Any] = load_pipeline_config(pipeline_path)

    # Parse pipeline config
    pipeline_name: str = pipeline_config.get('name', 'unnamed_pipeline')
    stages: list[dict[str, Any]] = pipeline_config.get('stages', [])
    shared_config: dict[str, Any] = pipeline_config.get('shared', {})

    # Validate pipeline config
    if not stages:
        print("Error: No stages defined in pipeline")
        return

    # Prepare output directory automatically if not specified
    if output_dir is None:
        #
        timestamp: str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #
        output_dir: str = f"trainings_pipeline/{timestamp}_{pipeline_name.replace(' ', '_')}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save pipeline config for reference
    shutil.copy(pipeline_path, os.path.join(output_dir, "pipeline.yaml"))

    # Print pipeline info
    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE: {pipeline_name}")
    print(f"Output: {output_dir}")
    print(f"Stages: {len(stages)}")
    print(f"{'='*60}\n")

    # Initialize weights path to None (for loading from previous stage)
    current_weights_path: Optional[str] = None

    # Iterate over stages
    #
    stage_idx: int
    stage: dict[str, Any]
    #
    for stage_idx, stage in enumerate(stages):

        # Get stage info
        stage_name: str = stage.get('name', f'Stage {stage_idx + 1}')
        stage_config_file: str = stage.get('config')
        max_episodes: int = stage.get('max_episodes', 10000)
        early_stop_successes: int = stage.get('early_stop_successes', 0)

        # Print stage info
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage_name}")
        print(f"Config: {stage_config_file}")
        print(f"Max episodes: {max_episodes}")
        #
        if early_stop_successes > 0:
            #
            print(f"Early stop after: {early_stop_successes} consecutive successes")
        #
        print(f"{'='*60}\n")

        # Load stage config
        stage_config_path: Path = pipeline_dir / stage_config_file
        #
        if not stage_config_path.exists():
            print(f"Error: Stage config not found: {stage_config_path}")
            return

        # Load stage config
        with open(stage_config_path, 'r') as f:
            stage_yaml: dict[str, Any] = yaml.safe_load(f)

        # Apply shared config (pipeline-level overrides)
        if shared_config:
            stage_yaml = merge_configs(stage_yaml, {'model': shared_config})

        # Override training parameters
        if 'training' not in stage_yaml:
            #
            stage_yaml['training'] = {}

        # Set max episodes
        stage_yaml['training']['max_episodes'] = max_episodes

        # Set early stopping
        if early_stop_successes > 0:
            stage_yaml['training']['early_stopping_enabled'] = True
            stage_yaml['training']['early_stopping_consecutive_successes'] = early_stop_successes

        # Load weights from previous stage
        if current_weights_path is not None:
            #
            stage_yaml['training']['load_weights_from'] = current_weights_path
            #
            print(f"Loading weights from: {current_weights_path}")

        # Set output path for this stage
        stage_output_dir: str = os.path.join(output_dir, f"stage_{stage_idx + 1:02d}_{stage_name.replace(' ', '_')}")
        os.makedirs(stage_output_dir, exist_ok=True)

        # Save merged config
        merged_config_path: str = os.path.join(stage_output_dir, "config.yaml")
        with open(merged_config_path, 'w') as f:
            yaml.dump(stage_yaml, f, default_flow_style=False)

        # Run training
        print(f"\nStarting training for stage {stage_idx + 1}...")

        try:
            # Load config
            config: GlobalConfig = load_config(merged_config_path)
            # Run training
            train(config, exp_dir_override=stage_output_dir)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print(f"Progress saved to: {stage_output_dir}")
            return

        except Exception as e:
            #
            print(f"\nError in training stage {stage_idx + 1}: {e}")
            #
            traceback.print_exc()
            #
            return

        # Update weights path for next stage
        best_model_path: str = os.path.join(stage_output_dir, "best_model.pth")
        #
        if os.path.exists(best_model_path):
            #
            current_weights_path = best_model_path
            #
            print(f"\nStage {stage_idx + 1} complete. Best model: {best_model_path}")
        else:
            print(f"\nWarning: No best model found for stage {stage_idx + 1}")

    # Print pipeline info
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"Final model: {current_weights_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


# Main function
def main():

    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Run training pipeline')
    #
    parser.add_argument('--pipeline', type=str, required=True,
                        help='Path to pipeline.yaml config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory')
    #
    args: argparse.Namespace = parser.parse_args()

    # Check if pipeline config exists
    if not os.path.exists(args.pipeline):
        print(f"Error: Pipeline config not found: {args.pipeline}")
        return

    # Run pipeline
    run_pipeline(args.pipeline, args.output)


# This script can also be directly run from the command line instead of using src.main
if __name__ == "__main__":
    #
    main()
