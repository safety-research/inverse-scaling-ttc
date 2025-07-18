"""
Creator for GPA regression synthetic dataset.
"""

import pandas as pd
import random
from pathlib import Path

from ..base.base_creator import BaseCreator
from ..base.data_models import TaskInstance


class GradesRegressionCreator(BaseCreator):
    """
    Creator for student lifestyle regression task.
    
    Generates few-shot learning prompts predicting GPA, stress level, or gender
    from student lifestyle metrics.
    """

    def __init__(self, config):
        super().__init__(config)
        self.data = None

    def setup_generation(self) -> None:
        """Load and prepare the student lifestyle dataset."""
        # Get input file from config or use default
        input_file = self.generation_config.custom_params.get(
            "input_file", "student_lifestyle_regression_dataset.csv"
        )

        input_path = Path(input_file)
        if not input_path.exists():
            # Try relative to project root
            input_path = Path(__file__).parent.parent.parent.parent / input_file

        if not input_path.exists():
            raise FileNotFoundError(f"Student lifestyle dataset not found: {input_file}")

        # Load and subsample data
        full_data = pd.read_csv(input_path)
        self.data = full_data.sample(n=min(500, len(full_data)), random_state=42).reset_index(drop=True)

        self.logger.info(f"Loaded student lifestyle dataset with {len(self.data)} samples")

    def generate_instance(self, instance_id: int) -> TaskInstance:
        """
        Generate a single GPA regression instance.
        
        Args:
            instance_id: Unique identifier for this instance
            
        Returns:
            TaskInstance with GPA regression problem
        """
        # Get configuration
        examples_per_prompt_options = self.generation_config.custom_params.get(
            "examples_per_prompt", [0, 8, 16, 32, 64]
        )
        answer_column = self.generation_config.custom_params.get("answer_column", "Grades")

        # Cycle through different numbers of examples
        examples_per_prompt = examples_per_prompt_options[instance_id % len(examples_per_prompt_options)]

        # Select target record
        target_idx = instance_id % len(self.data)
        target_record = self.data.iloc[target_idx]

        # Select example records (excluding target)
        available_indices = [i for i in range(len(self.data)) if i != target_idx]
        example_indices = random.sample(available_indices, min(examples_per_prompt, len(available_indices)))

        # Build prompt
        prompt = ""

        if examples_per_prompt > 0:
            prompt += "Here are some records about student lifestyle. These records are from students of university collected via a Google Form survey. It includes information on study hours, extracurricular activities, sleep, socializing, physical activity. The data comes from an academic year from August 2023 to May 2024 and reflects a student of Lisboa."
            
            for idx in example_indices:
                record = self.data.iloc[idx]
                prompt += "<record>\n"
                for col in record.index:
                    prompt += f"\t<{col}>{record[col]}</{col}>\n"
                prompt += "</record>\n"

        prompt += f"Based on the following information about a student, please predict the value of **{answer_column}** for the student between 0 and 10. Just respond with '<answer>...</answer>', and say nothing else.\n"
        
        # Add target record (excluding the answer column)
        prompt += "<record>\n"
        for col in target_record.index:
            if col != answer_column:
                prompt += f"\t<{col}>{target_record[col]}</{col}>\n"
        prompt += "</record>\n"
        answer = target_record[answer_column]

        # Convert answer to appropriate type
        if answer_column in ["Grades"]:
            answer = float(answer)
        elif answer_column in ["Stress_Level"]:
            answer = int(answer)
        else:
            answer = str(answer)

        # Create metadata
        metadata = {
            "dataset": "gpa_regression",
            "answer_column": answer_column,
            "examples_per_prompt": examples_per_prompt,
            "target_index": target_idx,
            "example_indices": example_indices
        }

        return TaskInstance(
            prompt=prompt,
            answer=answer,
            metadata=metadata
        )

