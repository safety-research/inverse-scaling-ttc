#!/usr/bin/env python3
"""
Generate datasets using the InverseScalingGym framework.

This script provides a command-line interface to generate synthetic datasets
or convert existing datasets using any registered task in the InverseScalingGym.

Examples:
    # Generate synthetic datasets
    python scripts/generate_dataset.py misleading_alignment --num-instances 1000 --seed 42
    python scripts/generate_dataset.py wordle --num-instances 500 --output-dir data/generated/
    
    # Convert existing datasets  
    python scripts/generate_dataset.py --convert bbeh --input data/bbeh/ --num-instances 1000
    
    # List available tasks
    python scripts/generate_dataset.py --list-tasks
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Any

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gym import InverseScalingGym
from src.gym.base.data_models import OutputFormat


def list_available_tasks(gym: InverseScalingGym) -> None:
    """List all available tasks with descriptions."""
    tasks = gym.list_available_tasks()
    
    print("🎯 Available Creator Tasks:")
    print("=" * 50)
    for task_name in sorted(tasks["creators"]):
        info = gym.get_task_info(task_name)
        if info:
            print(f"  {task_name:25} - {info['description']}")
            print(f"  {'':25}   Tags: {', '.join(info['tags'])}")
            print()
    
    print("\n📊 Available Converter Tasks:")
    print("=" * 50)
    for task_name in sorted(tasks["converters"]):
        info = gym.get_task_info(task_name)
        if info:
            print(f"  {task_name:25} - {info['description']}")
            print(f"  {'':25}   Tags: {', '.join(info['tags'])}")
            print()
    
    print(f"\nTotal: {len(tasks['creators'])} creators, {len(tasks['converters'])} converters")


def generate_dataset(
    gym: InverseScalingGym,
    task_name: str,
    num_instances: int,
    random_seed: int,
    output_format: str,
    output_dir: Path,
    **kwargs
) -> None:
    """Generate a synthetic dataset."""
    print(f"🎯 Generating dataset: {task_name}")
    print(f"   Instances: {num_instances}")
    print(f"   Random seed: {random_seed}")
    print(f"   Output format: {output_format}")
    
    # Determine output filename
    output_file = output_dir / f"{task_name}.jsonl"
    
    start_time = time.time()
    
    try:
        # Generate and save dataset
        metadata = gym.create_and_save(
            task_name=task_name,
            output_path=output_file,
            num_instances=num_instances,
            random_seed=random_seed,
            output_format=output_format,
            **kwargs
        )
        
        generation_time = time.time() - start_time
        file_size_mb = metadata.file_size_bytes / (1024 * 1024) if metadata.file_size_bytes else 0
        
        print(f"✅ Success!")
        print(f"   Generated: {metadata.total_instances} instances")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Output: {output_file} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        sys.exit(1)


def convert_dataset(
    gym: InverseScalingGym,
    task_name: str,
    input_source: str,
    output_format: str,
    output_dir: Path,
    num_instances: Optional[int] = None,
    **kwargs
) -> None:
    """Convert an existing dataset."""
    print(f"📊 Converting dataset: {task_name}")
    print(f"   Input: {input_source}")
    print(f"   Output format: {output_format}")
    
    # Determine output filename
    input_path = Path(input_source)
    if input_path.is_file():
        # If input is a single file, use its name in the output
        base_name = input_path.stem  # Get filename without extension
        output_file = output_dir / f"{base_name}.jsonl"
    else:
        # If input is a directory, use the task name
        output_file = output_dir / f"{task_name}.jsonl"
    
    start_time = time.time()
    
    try:
        # Convert and save dataset
        metadata = gym.convert_and_save(
            task_name=task_name,
            input_source=input_source,
            output_path=output_file,
            output_format=output_format,
            **kwargs
        )
        
        conversion_time = time.time() - start_time
        file_size_mb = metadata.file_size_bytes / (1024 * 1024) if metadata.file_size_bytes else 0
        
        # Truncate if requested
        if num_instances and metadata.total_instances > num_instances:
            print(f"🔄 Truncating to {num_instances} instances...")
            
            # Read back the file and truncate
            instances = []
            with open(output_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= num_instances:
                        break
                    instances.append(line.strip())
            
            # Write truncated version
            with open(output_file, 'w') as f:
                for instance in instances:
                    f.write(instance + '\n')
            
            metadata.total_instances = len(instances)
            metadata.file_size_bytes = output_file.stat().st_size
            file_size_mb = metadata.file_size_bytes / (1024 * 1024)
        
        print(f"✅ Success!")
        print(f"   Converted: {metadata.total_instances} instances")
        print(f"   Time: {conversion_time:.2f}s")
        print(f"   Output: {output_file} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Error converting dataset: {e}")
        sys.exit(1)


def _parse_custom_param_value(value: str) -> Any:
    """
    Parse a custom parameter value to appropriate Python type.
    
    Args:
        value: String value from command line
        
    Returns:
        Parsed value (int, float, bool, list, dict, or string)
    """
    # Try to parse as JSON first (handles lists, dicts, booleans, numbers)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    
    # Try to parse as int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try to parse as float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try to parse as boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    
    # Return as string
    return value


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate or convert datasets using InverseScalingGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main action arguments
    parser.add_argument(
        "task_name",
        nargs="?",
        help="Name of the task to run (use --list-tasks to see available tasks)"
    )
    
    # Action selection
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit"
    )
    
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert an existing dataset instead of generating synthetic data"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-instances",
        type=int,
        default=2500,
        help="Number of instances to generate (default: 2500)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["simple_qa", "multiple_choice", "yes_no", "classification"],
        default="simple_qa",
        help="Output format for the dataset (default: simple_qa)"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        help="Input source for conversion tasks (required for --convert)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Output directory for generated files (default: data/generated)"
    )
    
    # Additional task-specific parameters
    parser.add_argument(
        "--max-distractors",
        type=int,
        help="Maximum number of distractors for misleading tasks"
    )
    
    parser.add_argument(
        "--examples-per-prompt",
        type=int,
        nargs="+",
        help="Number of examples per prompt for few-shot tasks (can specify multiple)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Support for arbitrary custom parameters
    parser.add_argument(
        "--custom-param",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Custom parameter for task configuration (can be used multiple times). Example: --custom-param integer_ratio 0.7"
    )
    
    args = parser.parse_args()
    
    # Initialize gym
    gym = InverseScalingGym()
    
    # Handle list tasks
    if args.list_tasks:
        list_available_tasks(gym)
        return
    
    # Validate required arguments
    if not args.task_name:
        parser.error("task_name is required (or use --list-tasks)")
    
    if args.convert and not args.input:
        parser.error("--input is required when using --convert")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare additional kwargs
    kwargs = {}
    if args.max_distractors is not None:
        kwargs["max_distractors"] = args.max_distractors
    if args.examples_per_prompt is not None:
        kwargs["examples_per_prompt_options"] = args.examples_per_prompt
    
    # Parse custom parameters
    custom_params = {}
    if args.custom_param:
        for key, value in args.custom_param:
            # Try to parse value as appropriate type
            parsed_value = _parse_custom_param_value(value)
            custom_params[key] = parsed_value
        kwargs["custom_params"] = custom_params
    
    # Validate task exists
    available_tasks = gym.list_available_tasks()
    all_tasks = available_tasks["creators"] + available_tasks["converters"]
    
    if args.task_name not in all_tasks:
        print(f"❌ Error: Task '{args.task_name}' not found.")
        print(f"Available tasks: {', '.join(sorted(all_tasks))}")
        print("Use --list-tasks for detailed information.")
        sys.exit(1)
    
    # Execute requested action
    if args.convert:
        convert_dataset(
            gym=gym,
            task_name=args.task_name,
            input_source=args.input,
            output_format=args.output_format,
            output_dir=args.output_dir,
            num_instances=args.num_instances,
            **kwargs
        )
    else:
        generate_dataset(
            gym=gym,
            task_name=args.task_name,
            num_instances=args.num_instances,
            random_seed=args.seed,
            output_format=args.output_format,
            output_dir=args.output_dir,
            **kwargs
        )


if __name__ == "__main__":
    main()