from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore


@dataclass
class PathConfig:
    output_base_dir: str = "results"
    models_config: str = "config/model"
    tasks_config: str = "config/task"


@dataclass
class ModelConfig:
    id: str
    name: str
    model_name: str
    type: str
    system_prompt: str
    temperature: float = 0.0
    concurrency_limit: Optional[int] = None


@dataclass
class TaskConfig:
    name: str
    description: str
    category: str
    file_path: str
    metric: str = "accuracy"


@dataclass
class EvaluationConfig:
    models: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    reasoning_budgets: List[int] = field(default_factory=lambda: [0, 1024, 2048, 4096])
    batch_size: int = 100
    thinking: bool = False
    shot_count: Optional[int] = None
    sample_limit: Optional[int] = None
    seeds: Optional[List[int]] = None  # Explicit seeds for each run (VLLM models only)


@dataclass
class ApiConfig:
    use_batch_api: bool = True
    anthropic_api_key_tag: str = "ANTHROPIC_API_KEY_NORMAL_BATCH"
    no_cache: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = False
    samples: int = 20
    runs: int = 3
    seed: int = 42


@dataclass
class AnalysisConfig:
    run_analysis: bool = True
    plot_results: bool = True
    save_plots: bool = True
    plot_by_category: bool = True


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "inverse-scaling-eval"
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    name: str = "base"
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: Optional[ValidationConfig] = None


@dataclass
class Config:
    """Main configuration class that holds all sub-configurations"""
    paths: PathConfig = field(default_factory=PathConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def register_config_schemas():
    """Register the configuration schemas with Hydra's ConfigStore"""
    cs = ConfigStore.instance()

    # Register config classes
    cs.store(name="path", node=PathConfig)
    cs.store(name="model", node=ModelConfig)
    cs.store(name="task", node=TaskConfig)
    cs.store(name="evaluation", node=EvaluationConfig)
    cs.store(name="api", node=ApiConfig)
    cs.store(name="validation", node=ValidationConfig)
    cs.store(name="analysis", node=AnalysisConfig)
    cs.store(name="wandb", node=WandbConfig)
    cs.store(name="experiment", node=ExperimentConfig)
    cs.store(name="config", node=Config)
