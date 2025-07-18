"""
Converter for Big-Bench Extra Hard (BBEH) datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ..base.base_converter import BaseConverter
from ..base.data_models import TaskInstance, OutputFormat


class BBEHConverter(BaseConverter):
    """
    Converter for Big-Bench Extra Hard dataset format.
    
    Converts BBEH task.json files containing examples to standardized JSONL format.
    """

    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load BBEH data from directory structure.
        
        Expected structure:
        input_source/
        ├── task1/
        │   └── task.json
        ├── task2/
        │   └── task.json
        └── ...
        
        Yields:
            Raw data dictionaries with task name and examples
        """
        input_path = Path(self.config.input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        # If input_source points to a specific task.json file
        if input_path.is_file() and input_path.name == "task.json":
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                task_name = input_path.parent.name

            for example in data.get("examples", []):
                yield {
                    "task_name": task_name,
                    "example": example
                }

        # If input_source points to a directory containing task subdirectories
        elif input_path.is_dir():
            for task_dir in input_path.iterdir():
                if not task_dir.is_dir():
                    continue

                task_json = task_dir / "task.json"
                if not task_json.exists():
                    self.logger.warning(f"No task.json found in {task_dir}")
                    continue

                try:
                    with open(task_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for example in data.get("examples", []):
                        yield {
                            "task_name": task_dir.name,
                            "example": example
                        }

                except Exception as e:
                    self.logger.error(f"Error loading {task_json}: {e}")

        else:
            raise ValueError(f"Input source must be a directory or task.json file: {input_path}")

    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single BBEH example to TaskInstance.
        
        Args:
            raw_data: Dictionary containing task_name and example
            
        Returns:
            TaskInstance with converted data
        """
        example = raw_data["example"]
        task_name = raw_data["task_name"]

        # Extract input and target
        prompt = example.get("input", "")
        answer = example.get("target", "")

        if not prompt or not answer:
            self.logger.warning(f"Missing input or target in {task_name}")
            return None

        # Create metadata
        metadata = {
            "task_name": task_name,
            "original_data": example
        }

        # Handle different output formats
        if self.config.output_format == OutputFormat.MULTIPLE_CHOICE:
            # Check if this is a multiple choice question
            target_scores = example.get("target_scores", {})
            if target_scores:
                # Extract choices and find correct answer
                classes = list(target_scores.keys())
                # Find the choice with highest score (should be 1.0 for correct answer)
                correct_choice = max(target_scores.items(), key=lambda x: x[1])[0]
                answer_index = classes.index(correct_choice)

                return TaskInstance(
                    prompt=prompt,
                    answer=correct_choice,
                    answer_index=answer_index,
                    classes=classes,
                    metadata=metadata
                )
            else:
                # Not a multiple choice question, treat as simple QA
                return TaskInstance(
                    prompt=prompt,
                    answer=answer,
                    metadata=metadata
                )
        else:
            # Simple QA format
            return TaskInstance(
                prompt=prompt,
                answer=answer,
                metadata=metadata
            )
