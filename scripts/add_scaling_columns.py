# This script is designed to be run from the root of the project.
# Helper script to add scaling columns to the prediction files.


import json
import os
import shutil
from pathlib import Path


# Define the original data paths for the main tasks.
ORIGINAL_DATA_PATHS = {
    "student_lifestyle_regression_Grades": Path("data/new_tasks/student_lifestyle_regression_Grades.jsonl"),
    "student_lifestyle_regression_Grades_simpson_paradox": Path("data/new_tasks/student_lifestyle_regression_Grades_simpson_paradox.jsonl"),
    "synthetic_misleading_math": Path("data/new_tasks/synthetic-misleading-math-5-distractors.jsonl"),
    "synthetic_misleading_python": Path("data/new_tasks/synthetic-misleading-python-code-5-distractors.jsonl"),
    "synthetic_misleading_alignment": Path("data/new_tasks/synthetic-misleading-alignment-5-distractors.jsonl"),
    "synthetic_misleading_philosophy": Path("data/new_tasks/synthetic-misleading-philosophy-5-distractors.jsonl"),
    "synthetic_misleading_cognitive_bias": Path("data/new_tasks/synthetic-misleading-cognitive-biases-5-distractors.jsonl"),
    "synthetic_misleading_math_famous_paradoxes": Path("data/new_tasks/synthetic_misleading_math_famous_paradoxes.jsonl"),
    "bbeh_zebra_puzzles": Path("data/new_tasks/bbeh_zebra_puzzles.jsonl"),
}

# Define the scaling columns for each task.
SCALING_COLUMNS = {
    "student_lifestyle_regression_Grades": "examples_per_prompt",
    "student_lifestyle_regression_Grades_simpson_paradox": "sampling_strategy",
    "synthetic_misleading_math": "num_distractors",
    "synthetic_misleading_python": "num_distractors",
    "synthetic_misleading_alignment": "num_distractors",
    "synthetic_misleading_philosophy": "num_distractors",
    "synthetic_misleading_cognitive_bias": "num_distractors",
    "synthetic_misleading_math_famous_paradoxes": "num_distractors",
    "bbeh_zebra_puzzles": "grid_size",
    "synthetic_scheduling": "target_clue_count",
    "synthetic_smith_jones_robinson": "target_clue_count",
}

# Define the input and output directories.
PREDICTIONS_BASE_DIR = Path("results_v2")
OUTPUT_BASE_DIR = Path("results_v2_scaled")

def add_scaling_columns():
    """
    Adds scaling columns to the prediction files from the original data files.
    """
    print("🚀 Starting to add scaling columns to prediction files...")

    if not PREDICTIONS_BASE_DIR.is_dir():
        print(f"❌ Predictions directory not found: {PREDICTIONS_BASE_DIR}")
        return

    for model_dir in PREDICTIONS_BASE_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        if model_dir.name not in ["qwen3_32b_natural_overthinking", "qwq_32b_natural_overthinking"]:
            continue

        print(f"Processing model: {model_dir.name}")
        output_model_dir = OUTPUT_BASE_DIR / model_dir.name
        output_model_dir.mkdir(parents=True, exist_ok=True)

        for task_id, original_data_path in ORIGINAL_DATA_PATHS.items():
            prediction_file = model_dir / f"{task_id}.jsonl"
            output_file = output_model_dir / f"{task_id}.jsonl"

            if not prediction_file.exists():
                print(f"  - Skipping {task_id}: Prediction file not found.")
                continue

            if not original_data_path.exists():
                print(f"  - Skipping {task_id}: Original data file not found at {original_data_path}.")
                continue
            
            print(f"  - Processing task: {task_id}")
            
            original_data = [json.loads(line) for line in original_data_path.read_text().strip().split('\n')]
            predictions = [json.loads(line) for line in prediction_file.read_text().strip().split('\n')]

            if not predictions:
                print(f"    - Warning: No predictions found for {task_id}.")
                continue

            scaling_column = SCALING_COLUMNS.get(task_id)
            if not scaling_column:
                print(f"    - Warning: No scaling column defined for {task_id}. Copying file as is.")
                shutil.copy(prediction_file, output_file)
                continue

            scaling_map = {
                f"{task_id}_{i}": item.get(scaling_column)
                for i, item in enumerate(original_data)
            }

            new_predictions = []
            missing_ids = 0
            unmatched_ids = 0
            for p in predictions:
                instance_id = p.get("instance_id")

                if "_run" in instance_id:
                    instance_id = instance_id.split("_run")[0]

                instance_id = instance_id.replace("-", "_")

                if not instance_id:
                    missing_ids += 1
                elif instance_id in scaling_map:
                    p[scaling_column] = scaling_map.get(instance_id)
                else:
                    unmatched_ids += 1
                new_predictions.append(p)

            if missing_ids > 0:
                print(
                    f"    - Warning: {missing_ids} predictions in {task_id} are missing 'instance_id'."
                )
            if unmatched_ids > 0:
                print(
                    f"    - Warning: Could not find matching original data for {unmatched_ids} predictions in {task_id}."
                )

            with output_file.open("w") as f:
                for p in new_predictions:
                    f.write(json.dumps(p) + '\n')

    print("✅ Successfully added scaling columns.")

if __name__ == "__main__":
    add_scaling_columns() 