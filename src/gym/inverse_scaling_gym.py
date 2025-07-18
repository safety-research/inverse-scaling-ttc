"""
Main InverseScalingGym class for orchestrating dataset generation and conversion.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base.data_models import (
    TaskConfig, TaskInstance, TaskMetadata, GenerationConfig,
    TaskType, OutputFormat
)
from .base.task_registry import TaskRegistry, get_global_registry

logger = logging.getLogger(__name__)


class InverseScalingGym:
    """
    Main interface for the InverseScalingGym framework.
    
    This class provides a unified interface for creating synthetic datasets
    and converting existing datasets to the standardized JSONL format.
    """

    def __init__(self, registry: Optional[TaskRegistry] = None):
        """
        Initialize the InverseScalingGym.
        
        Args:
            registry: Optional custom task registry (uses global if not provided)
        """
        self.registry = registry or get_global_registry()
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_task(
        self,
        task_name: str,
        num_instances: int = 500,
        random_seed: int = 42,
        output_format: Union[str, OutputFormat] = OutputFormat.SIMPLE_QA,
        **kwargs
    ) -> List[TaskInstance]:
        """
        Create a synthetic dataset using a registered creator task.
        
        Args:
            task_name: Name of the registered creator task
            num_instances: Number of instances to generate
            random_seed: Random seed for reproducibility
            output_format: Output format for the task instances
            **kwargs: Additional parameters for the generation config
            
        Returns:
            List of generated TaskInstances
            
        Raises:
            KeyError: If task not found
            ValueError: If task is not a creator or configuration is invalid
        """
        # Ensure output_format is OutputFormat enum
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)

        # Create generation config
        generation_config = GenerationConfig(
            num_instances=num_instances,
            random_seed=random_seed,
            output_format=output_format,
            **kwargs
        )

        # Get default config and update with generation config
        default_config = self.registry.get_config(task_name)
        if default_config is None:
            # Create a basic config
            config = TaskConfig(
                name=task_name,
                task_type=TaskType.CREATE,
                description=f"Synthetic task: {task_name}",
                output_format=output_format,
                generation_config=generation_config
            )
        else:
            # Update the default config
            config = TaskConfig(
                name=default_config.name,
                task_type=default_config.task_type,
                description=default_config.description,
                input_source=default_config.input_source,
                input_format=default_config.input_format,
                output_format=output_format,
                output_filename=default_config.output_filename,
                generation_config=generation_config,
                tags=default_config.tags,
                inverse_scaling_properties=default_config.inverse_scaling_properties,
                expected_trends=default_config.expected_trends
            )

        # Validate this is a creator task
        if task_name not in self.registry.list_creators():
            raise ValueError(f"Task '{task_name}' is not a creator task")

        # Create and run the task
        start_time = time.time()
        creator = self.registry.create_task(task_name, config)
        instances = creator.generate_dataset()
        generation_time = time.time() - start_time

        self.logger.info(
            f"Generated {len(instances)} instances for task '{task_name}' "
            f"in {generation_time:.2f} seconds"
        )

        return instances

    def convert_task(
        self,
        task_name: str,
        input_source: Union[str, Path],
        output_format: Union[str, OutputFormat] = OutputFormat.SIMPLE_QA,
        **kwargs
    ) -> List[TaskInstance]:
        """
        Convert an existing dataset using a registered converter task.
        
        Args:
            task_name: Name of the registered converter task
            input_source: Path to the source data
            output_format: Output format for the task instances
            **kwargs: Additional parameters for the task config
            
        Returns:
            List of converted TaskInstances
            
        Raises:
            KeyError: If task not found
            ValueError: If task is not a converter or configuration is invalid
        """
        # Ensure output_format is OutputFormat enum
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)

        # Get default config and update with provided parameters
        default_config = self.registry.get_config(task_name)
        if default_config is None:
            # Create a basic config
            config = TaskConfig(
                name=task_name,
                task_type=TaskType.CONVERT,
                description=f"Conversion task: {task_name}",
                input_source=input_source,
                output_format=output_format,
                **kwargs
            )
        else:
            # Update the default config
            config = TaskConfig(
                name=default_config.name,
                task_type=default_config.task_type,
                description=default_config.description,
                input_source=input_source,
                input_format=default_config.input_format,
                output_format=output_format,
                output_filename=default_config.output_filename,
                generation_config=default_config.generation_config,
                tags=default_config.tags,
                inverse_scaling_properties=default_config.inverse_scaling_properties,
                expected_trends=default_config.expected_trends
            )

        # Validate this is a converter task
        if task_name not in self.registry.list_converters():
            raise ValueError(f"Task '{task_name}' is not a converter task")

        # Create and run the task
        start_time = time.time()
        converter = self.registry.create_task(task_name, config)
        instances = converter.convert_dataset()
        conversion_time = time.time() - start_time

        self.logger.info(
            f"Converted {len(instances)} instances for task '{task_name}' "
            f"in {conversion_time:.2f} seconds"
        )

        return instances

    def save_instances_to_file(
        self,
        instances: List[TaskInstance],
        output_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> TaskMetadata:
        """
        Save task instances to a JSONL file.
        
        Args:
            instances: List of task instances to save
            output_path: Path where to save the JSONL file
            metadata: Optional metadata to include in a separate file
            
        Returns:
            TaskMetadata with information about the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save instances as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in instances:
                json.dump(instance.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        file_size = output_path.stat().st_size

        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix('.meta.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(instances)} instances to {output_path}")

        # Create basic metadata
        task_metadata = TaskMetadata(
            total_instances=len(instances),
            output_format=OutputFormat.SIMPLE_QA,  # Default, should be passed from caller
            generation_time=0.0,  # Should be set by caller
            config=TaskConfig(  # Minimal config
                name="unknown",
                task_type=TaskType.CREATE,
                description="Generated via save_instances_to_file"
            ),
            file_size_bytes=file_size
        )

        return task_metadata

    def create_and_save(
        self,
        task_name: str,
        output_path: Union[str, Path],
        num_instances: int = 500,
        random_seed: int = 42,
        output_format: Union[str, OutputFormat] = OutputFormat.SIMPLE_QA,
        **kwargs
    ) -> TaskMetadata:
        """
        Create a synthetic dataset and save it to file in one operation.
        
        Args:
            task_name: Name of the registered creator task
            output_path: Path where to save the JSONL file
            num_instances: Number of instances to generate
            random_seed: Random seed for reproducibility
            output_format: Output format for the task instances
            **kwargs: Additional parameters for the generation config
            
        Returns:
            TaskMetadata with information about the generated task
        """
        start_time = time.time()

        # Generate instances
        instances = self.create_task(
            task_name=task_name,
            num_instances=num_instances,
            random_seed=random_seed,
            output_format=output_format,
            **kwargs
        )

        # Save to file
        metadata = self.save_instances_to_file(instances, output_path)

        # Update metadata with actual generation time and format
        generation_time = time.time() - start_time
        metadata.generation_time = generation_time
        metadata.output_format = OutputFormat(output_format) if isinstance(output_format, str) else output_format

        return metadata

    def convert_and_save(
        self,
        task_name: str,
        input_source: Union[str, Path],
        output_path: Union[str, Path],
        output_format: Union[str, OutputFormat] = OutputFormat.SIMPLE_QA,
        **kwargs
    ) -> TaskMetadata:
        """
        Convert an existing dataset and save it to file in one operation.
        
        Args:
            task_name: Name of the registered converter task
            input_source: Path to the source data
            output_path: Path where to save the JSONL file
            output_format: Output format for the task instances
            **kwargs: Additional parameters for the task config
            
        Returns:
            TaskMetadata with information about the converted task
        """
        start_time = time.time()

        # Convert instances
        instances = self.convert_task(
            task_name=task_name,
            input_source=input_source,
            output_format=output_format,
            **kwargs
        )

        # Save to file
        metadata = self.save_instances_to_file(instances, output_path)

        # Update metadata with actual conversion time and format
        conversion_time = time.time() - start_time
        metadata.generation_time = conversion_time
        metadata.output_format = OutputFormat(output_format) if isinstance(output_format, str) else output_format

        return metadata

    def list_available_tasks(self) -> Dict[str, List[str]]:
        """
        List all available tasks organized by type.
        
        Returns:
            Dictionary with 'converters' and 'creators' keys
        """
        return {
            "converters": self.registry.list_converters(),
            "creators": self.registry.list_creators()
        }

    def get_task_info(self, task_name: str) -> Optional[Dict]:
        """
        Get information about a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with task information or None if not found
        """
        config = self.registry.get_config(task_name)
        if config is None:
            return None

        return {
            "name": config.name,
            "type": config.task_type.value,
            "description": config.description,
            "output_format": config.output_format.value,
            "tags": config.tags,
            "inverse_scaling_properties": config.inverse_scaling_properties,
            "expected_trends": config.expected_trends
        }
