"""
InverseScalingGym: A unified framework for creating and converting evaluation datasets.

This package provides tools for:
1. Converting existing datasets to standardized JSONL format
2. Creating synthetic datasets with inverse scaling properties
3. Managing task configurations and metadata
"""

from .inverse_scaling_gym import InverseScalingGym
from .base.task_registry import TaskRegistry
from .base.data_models import TaskInstance, TaskConfig, GenerationConfig

# Import register_tasks to trigger automatic registration
from . import register_tasks

__all__ = [
    "InverseScalingGym",
    "TaskRegistry",
    "TaskInstance",
    "TaskConfig",
    "GenerationConfig"
]
