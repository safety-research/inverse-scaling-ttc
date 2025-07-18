"""
Data models and schemas for the InverseScalingGym framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks supported by the framework."""
    CONVERT = "convert"  # Convert existing datasets
    CREATE = "create"    # Create synthetic datasets


class OutputFormat(str, Enum):
    """Supported output formats for tasks."""
    SIMPLE_QA = "simple_qa"           # {"prompt": str, "answer": str}
    MULTIPLE_CHOICE = "multiple_choice"  # {"prompt": str, "classes": [str], "answer_index": int}
    YES_NO = "yes_no"                 # {"prompt": str, "answer": "Yes"/"No"}
    CLASSIFICATION = "classification"  # {"prompt": str, "classes": [str], "answer_index": int}
    REGRESSION = "regression"         # {"prompt": str, "answer": float}


@dataclass
class TaskInstance:
    """A single task instance/example."""
    prompt: str
    answer: Union[str, int, float]
    answer_index: Optional[int] = None
    classes: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSONL output."""
        result = {
            "prompt": self.prompt,
            "answer": self.answer
        }

        if self.answer_index is not None:
            result["answer_index"] = self.answer_index

        if self.classes is not None:
            result["classes"] = self.classes

        # Add any additional metadata
        result.update(self.metadata)

        return result


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    num_instances: int = 500
    random_seed: int = 42
    output_format: OutputFormat = OutputFormat.SIMPLE_QA

    # Generation-specific parameters
    max_distractors: int = 5
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0

    # Few-shot learning parameters
    examples_per_prompt: List[int] = field(default_factory=lambda: [0])
    examples_per_prompt_options: List[int] = field(default_factory=lambda: [0, 8, 16, 32, 64])
    min_examples_per_prompt: int = 5
    max_examples_per_prompt: int = 15

    # Sequence generation parameters
    sequence_length_range: Tuple[int, int] = (5, 15)

    # Template and variation parameters
    templates: Optional[List[str]] = None
    variations: Optional[Dict[str, Any]] = None

    # Custom parameters for specific tasks
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    name: str
    task_type: TaskType
    description: str

    # Input configuration
    input_source: Optional[Union[str, Path]] = None
    input_format: Optional[str] = None

    # Output configuration
    output_format: OutputFormat = OutputFormat.SIMPLE_QA
    output_filename: Optional[str] = None

    # Processing configuration
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    # Task metadata
    tags: List[str] = field(default_factory=list)
    inverse_scaling_properties: List[str] = field(default_factory=list)
    expected_trends: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.task_type == TaskType.CONVERT and self.input_source is None:
            raise ValueError("Convert tasks must specify an input_source")

        if self.output_filename is None:
            self.output_filename = f"{self.name}.jsonl"


@dataclass
class ValidationResult:
    """Result of task validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMetadata:
    """Metadata about a generated or converted task."""
    total_instances: int
    output_format: OutputFormat
    generation_time: float
    config: TaskConfig
    validation_result: Optional[ValidationResult] = None
    file_size_bytes: Optional[int] = None
