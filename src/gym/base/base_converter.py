"""
Base class for dataset conversion tasks.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .data_models import TaskConfig, TaskInstance, TaskMetadata, ValidationResult

logger = logging.getLogger(__name__)


class BaseConverter(ABC):
    """
    Abstract base class for converting existing datasets to standardized format.
    
    All converter implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: TaskConfig):
        """
        Initialize the converter with a task configuration.
        
        Args:
            config: Task configuration containing input/output settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load data from the source location specified in config.
        
        Returns:
            Iterator yielding raw data dictionaries
            
        Raises:
            FileNotFoundError: If source data cannot be found
            ValueError: If source data format is invalid
        """
        pass

    @abstractmethod
    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single raw data instance to a TaskInstance.
        
        Args:
            raw_data: Raw data dictionary from source
            
        Returns:
            TaskInstance if conversion successful, None if should be skipped
            
        Raises:
            ValueError: If raw data cannot be converted
        """
        pass

    def filter_instance(self, instance: TaskInstance) -> bool:
        """
        Filter instances based on quality or other criteria.
        
        Args:
            instance: Converted task instance
            
        Returns:
            True if instance should be included, False otherwise
        """
        # Default implementation accepts all instances
        return True

    def validate_instance(self, instance: TaskInstance) -> ValidationResult:
        """
        Validate a converted instance for quality and correctness.
        
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

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def convert_dataset(self) -> List[TaskInstance]:
        """
        Convert the entire dataset from source to TaskInstance format.
        
        Returns:
            List of converted and validated TaskInstances
        """
        instances = []
        total_processed = 0
        skipped_count = 0
        invalid_count = 0

        self.logger.info(f"Starting conversion for task: {self.config.name}")

        try:
            for raw_data in self.load_source_data():
                total_processed += 1

                try:
                    # Convert the instance
                    instance = self.convert_instance(raw_data)
                    if instance is None:
                        skipped_count += 1
                        continue

                    # Filter the instance
                    if not self.filter_instance(instance):
                        skipped_count += 1
                        continue

                    # Validate the instance
                    validation = self.validate_instance(instance)
                    if not validation.is_valid:
                        invalid_count += 1
                        self.logger.warning(
                            f"Invalid instance at index {total_processed}: {validation.errors}"
                        )
                        continue

                    instances.append(instance)

                except Exception as e:
                    invalid_count += 1
                    self.logger.error(f"Error converting instance {total_processed}: {e}")

        except Exception as e:
            self.logger.error(f"Error loading source data: {e}")
            raise

        self.logger.info(
            f"Conversion complete. Processed: {total_processed}, "
            f"Valid: {len(instances)}, Skipped: {skipped_count}, Invalid: {invalid_count}"
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
