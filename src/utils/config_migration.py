"""
Utilities for migrating from the old config structure to the new Hydra-based structure.
These functions help maintain backward compatibility during the transition.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_old_style_config(config_path: str) -> Dict[str, Any]:
    """Load a config file in the old flat format"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def build_model_configs_from_yaml(models_yaml_path: str, output_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Convert a flat models.yaml file to individual model config files for the new structure.
    
    Args:
        models_yaml_path: Path to the old models.yaml file
        output_dir: Directory to write the new model YAML files
    
    Returns:
        Dictionary mapping model IDs to their configurations
    """
    models_config = load_old_style_config(models_yaml_path)

    # If output directory specified, write individual model files
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        for model_id, model_config in models_config.items():
            # Ensure id is included in the config
            model_config['id'] = model_id

            # Add package directive
            final_config = {
                # Add the package directive as a comment
                # '@package': 'model',
                **model_config
            }

            # Create filename from model ID, replacing problematic characters
            filename = model_id.replace('/', '_').replace('-', '_').replace('.', '_')
            filepath = os.path.join(output_dir, f"{filename}.yaml")

            with open(filepath, 'w') as f:
                f.write("# @package model\n\n")  # Add as a comment line
                yaml.dump(model_config, f, default_flow_style=False)

            print(f"Created model config: {filepath}")

    return models_config

def build_task_configs_from_yaml(tasks_yaml_path: str, output_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Convert a flat tasks.yaml file to task config files for the new structure.
    
    Args:
        tasks_yaml_path: Path to the old tasks.yaml file
        output_dir: Directory to write the new task YAML files
    
    Returns:
        Dictionary mapping task IDs to their configurations
    """
    tasks_config = load_old_style_config(tasks_yaml_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Group tasks by category
        tasks_by_category = {}
        for task_id, task_config in tasks_config.items():
            category = task_config.get('category', 'default')
            if category not in tasks_by_category:
                tasks_by_category[category] = {}
            tasks_by_category[category][task_id] = task_config

        # Create YAML files for each category
        for category, tasks in tasks_by_category.items():
            category_filename = category.lower().replace(' ', '_')
            filepath = os.path.join(output_dir, f"{category_filename}_tasks.yaml")

            # Build the config
            config = {
                'tasks': list(tasks.keys()),
                'task_definitions': tasks
            }

            with open(filepath, 'w') as f:
                f.write("# @package task\n\n")
                yaml.dump(config, f, default_flow_style=False)

            print(f"Created task config: {filepath}")

    return tasks_config

def migrate_experiment_configs(experiment_dir: str, models_dir: str, tasks_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Migrate experiment configs to the new format that uses Hydra defaults and overrides.
    
    Args:
        experiment_dir: Directory containing old experiment YAML files
        models_dir: Directory containing the new model config files
        tasks_dir: Directory containing the new task config files
        output_dir: Directory to write the migrated experiment files (default: same as experiment_dir)
    """
    if output_dir is None:
        output_dir = experiment_dir

    os.makedirs(output_dir, exist_ok=True)

    # Get available model and task configs
    model_files = [p.stem for p in Path(models_dir).glob("*.yaml")]
    task_files = [p.stem for p in Path(tasks_dir).glob("*.yaml")]

    # Process each experiment file
    for exp_file in Path(experiment_dir).glob("*.yaml"):
        with open(exp_file, 'r') as f:
            old_config = yaml.safe_load(f)

        # Skip if already migrated (has @package or defaults)
        if '@package' in old_config or 'defaults' in old_config:
            print(f"Skipping already migrated config: {exp_file}")
            continue

        # Determine most appropriate model and task files to use as defaults
        models = old_config.get('evaluation', {}).get('models', [])
        tasks = old_config.get('evaluation', {}).get('tasks', [])

        model_file = None
        for m in model_files:
            if any(model_id.lower().replace('-', '_') in m.lower() for model_id in models):
                model_file = m
                break

        task_file = None
        task_categories = set()
        for task_id in tasks:
            if isinstance(task_id, str) and 'synthetic' in task_id:
                task_categories.add('synthetic')

        if 'synthetic' in task_categories:
            task_file = 'synthetic_tasks'
        else:
            task_file = 'default'

        # Create new migrated config
        new_config = {
            '@package': '_global_',
            'defaults': [
                f"override /model: {model_file}" if model_file else None,
                f"override /task: {task_file}" if task_file else None,
                '_self_'
            ],
            'experiment': {
                'name': old_config.get('name', exp_file.stem)
            }
        }

        # Remove None values from defaults
        new_config['defaults'] = [d for d in new_config['defaults'] if d is not None]

        # Move specific sections from old config to new one
        if 'evaluation' in old_config:
            new_config['evaluation'] = old_config['evaluation']

        if 'api' in old_config:
            new_config['api'] = old_config['api']

        if 'wandb' in old_config:
            new_config['wandb'] = old_config['wandb']

        if 'analysis' in old_config:
            new_config['analysis'] = old_config['analysis']

        # Write the new config
        output_path = Path(output_dir) / f"{exp_file.stem}_migrated.yaml"
        with open(output_path, 'w') as f:
            f.write("# @package _global_\n\n")
            yaml.dump(new_config, f, default_flow_style=False)

        print(f"Migrated experiment config: {output_path}")

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Migrate configuration files to the new Hydra structure")
    parser.add_argument("--models", required=True, help="Path to the old models.yaml file")
    parser.add_argument("--tasks", required=True, help="Path to the old tasks.yaml file")
    parser.add_argument("--experiments", required=True, help="Directory containing old experiment YAML files")
    parser.add_argument("--output-models", help="Directory to write the new model config files")
    parser.add_argument("--output-tasks", help="Directory to write the new task config files")
    parser.add_argument("--output-experiments", help="Directory to write the migrated experiment files")

    args = parser.parse_args()

    # Generate model configs
    build_model_configs_from_yaml(args.models, args.output_models)

    # Generate task configs
    build_task_configs_from_yaml(args.tasks, args.output_tasks)

    # Migrate experiment configs
    migrate_experiment_configs(
        args.experiments,
        args.output_models or os.path.dirname(args.models),
        args.output_tasks or os.path.dirname(args.tasks),
        args.output_experiments
    )
