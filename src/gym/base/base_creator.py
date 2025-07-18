"""
Base class for synthetic dataset creation tasks.
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_models import TaskConfig, TaskInstance, TaskMetadata, ValidationResult

logger = logging.getLogger(__name__)


class BaseCreator(ABC):
    """
    Abstract base class for creating synthetic datasets.
    
    All creator implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: TaskConfig):
        """
        Initialize the creator with a task configuration.
        
        Args:
            config: Task configuration containing generation settings
        """
        self.config = config
        self.generation_config = config.generation_config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Set random seed for reproducibility
        random.seed(self.generation_config.random_seed)

    @abstractmethod
    def generate_instance(self, instance_id: int) -> TaskInstance:
        """
        Generate a single synthetic task instance.
        
        Args:
            instance_id: Unique identifier for this instance (0-based)
            
        Returns:
            Generated TaskInstance
            
        Raises:
            ValueError: If instance cannot be generated
        """
        pass

    def setup_generation(self) -> None:
        """
        Setup method called before generation begins.
        Override to perform any initialization (load templates, etc.)
        """
        pass

    def cleanup_generation(self) -> None:
        """
        Cleanup method called after generation completes.
        Override to perform any cleanup tasks.
        """
        pass

    def validate_instance(self, instance: TaskInstance) -> ValidationResult:
        """
        Validate a generated instance for quality and correctness.
        
        Args:
            instance: Task instance to validate
            
        Returns:
            ValidationResult with validation status and any issues
        """
        errors = []
        warnings = []

        # Basic validation
        if not instance.prompt or not instance.prompt.strip():
            errors.append("Empty or whitespace-only prompt")

        if instance.answer is None or (isinstance(instance.answer, str) and not instance.answer.strip()):
            errors.append("Missing or empty answer")

        # Format-specific validation
        if self.config.output_format.value == "multiple_choice":
            if not instance.classes:
                errors.append("Multiple choice task missing classes")
            elif instance.answer_index is None:
                errors.append("Multiple choice task missing answer_index")
            elif not (0 <= instance.answer_index < len(instance.classes)):
                errors.append("answer_index out of range for classes")

        # Check prompt length (reasonable bounds)
        if len(instance.prompt) > 10000:
            warnings.append("Very long prompt (>10k characters)")
        elif len(instance.prompt) < 10:
            warnings.append("Very short prompt (<10 characters)")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def generate_dataset(self) -> List[TaskInstance]:
        """
        Generate the complete synthetic dataset.
        
        Returns:
            List of generated and validated TaskInstances
        """
        instances = []
        invalid_count = 0

        self.logger.info(f"Starting generation for task: {self.config.name}")
        self.logger.info(f"Target instances: {self.generation_config.num_instances}")

        try:
            self.setup_generation()

            for instance_id in range(self.generation_config.num_instances):
                try:
                    # Generate the instance
                    instance = self.generate_instance(instance_id)

                    # Validate the instance
                    validation = self.validate_instance(instance)
                    if not validation.is_valid:
                        invalid_count += 1
                        self.logger.warning(
                            f"Invalid instance {instance_id}: {validation.errors}"
                        )
                        continue

                    # Log warnings if any
                    if validation.warnings:
                        self.logger.debug(
                            f"Instance {instance_id} warnings: {validation.warnings}"
                        )

                    instances.append(instance)

                except Exception as e:
                    invalid_count += 1
                    self.logger.error(f"Error generating instance {instance_id}: {e}")

        finally:
            self.cleanup_generation()

        self.logger.info(
            f"Generation complete. Valid: {len(instances)}, Invalid: {invalid_count}"
        )

        return instances

    def save_instances(self, instances: List[TaskInstance], output_path: Path) -> TaskMetadata:
        """
        Save task instances to JSONL file.
        
        Args:
            instances: List of task instances to save
            output_path: Path where to save the JSONL file
            
        Returns:
            TaskMetadata with information about the saved task
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in instances:
                json.dump(instance.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        file_size = output_path.stat().st_size

        self.logger.info(f"Saved {len(instances)} instances to {output_path}")

        return TaskMetadata(
            total_instances=len(instances),
            output_format=self.config.output_format,
            generation_time=0.0,  # Would be set by calling code
            config=self.config,
            file_size_bytes=file_size
        )

    # Utility methods for common generation patterns

    def random_choice_weighted(self, choices: List[Any], weights: Optional[List[float]] = None) -> Any:
        """
        Make a weighted random choice from a list.
        
        Args:
            choices: List of items to choose from
            weights: Optional weights for each choice
            
        Returns:
            Randomly selected item
        """
        if weights is None:
            return random.choice(choices)
        else:
            return random.choices(choices, weights=weights)[0]

    def generate_distractor(self, correct_answer: Any, distractor_type: str = "random") -> Any:
        """
        Generate a distractor/wrong answer based on the correct answer.
        
        Args:
            correct_answer: The correct answer to generate distractor for
            distractor_type: Type of distractor generation strategy
            
        Returns:
            Generated distractor
        """
        # Default implementation - override in subclasses for specific logic
        if isinstance(correct_answer, (int, float)):
            # Numeric distractor
            if distractor_type == "close":
                return correct_answer + random.choice([-1, 1]) * random.uniform(0.1, 0.5)
            else:
                return correct_answer * random.uniform(0.5, 2.0)
        elif isinstance(correct_answer, str):
            # String distractor - simple modification
            return f"not_{correct_answer}"
        else:
            return "distractor"

    def apply_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Apply variable substitution to a template string.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Template with variables substituted
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template missing variable: {e}")
        except Exception as e:
            self.logger.error(f"Template formatting error: {e}")
            raise ValueError(f"Template formatting error: {e}")
