"""
Task registration script for the InverseScalingGym.

This script registers all available converter and creator tasks with the global registry.
Add new task registrations here to make them available to the InverseScalingGym.
"""

from .base.task_registry import get_global_registry
from .base.data_models import TaskConfig, TaskType, OutputFormat, GenerationConfig

# Import all converter implementations
from .converters.asdiv_converter import ASDivConverter
from .converters.bbeh_converter import BBEHConverter
from .converters.gsm_ic_converter import GSMICConverter
from .converters.model_written_eval_converter import ModelWrittenEvalConverter
from .converters.multiarith_converter import MultiArithConverter

# Import all creator implementations
from .creators.misleading_math_creator import MisleadingMathCreator
from .creators.grades_regression_creator import GradesRegressionCreator
from .creators.misleading_python_creator import MisleadingPythonCreator
from .creators.misleading_alignment_creator import MisleadingAlignmentCreator
from .creators.misleading_cognitive_biases_creator import MisleadingCognitiveBiasesCreator
from .creators.misleading_game_theory_creator import MisleadingGameTheoryCreator
from .creators.misleading_philosophy_creator import MisleadingPhilosophyCreator


def register_all_tasks():
    """Register all available tasks with the global registry."""
    registry = get_global_registry()
    _register_converters(registry)
    _register_creators(registry)


def _register_converters(registry):
    """Register all converter tasks."""
    
    # ASDivConverter
    registry.register_converter("asdiv", ASDivConverter, TaskConfig(
        name="asdiv",
        task_type=TaskType.CONVERT,
        description="Convert ASDiv dataset from Hugging Face to standardized format",
        input_source="huggingface:MU-NLPC/Calc-ape_gap",
        input_format="huggingface",
        output_format=OutputFormat.SIMPLE_QA,
        tags=["math", "arithmetic", "huggingface"],
        inverse_scaling_properties=["arithmetic_reasoning"],
        expected_trends={"accuracy": "stable"}
    ))

    # BBEH Converter
    registry.register_converter("bbeh", BBEHConverter, TaskConfig(
        name="bbeh",
        task_type=TaskType.CONVERT,
        description="Convert Big-Bench Extra Hard dataset to standardized format",
        input_source="data/original/bbeh",
        input_format="bbeh_json",
        output_format=OutputFormat.SIMPLE_QA,
        tags=["benchmark", "reasoning", "hard"],
        inverse_scaling_properties=["complex_reasoning"],
        expected_trends={"accuracy": "decreasing"}
    ))

    # GSM-IC Converter
    registry.register_converter("gsm_ic", GSMICConverter, TaskConfig(
        name="gsm_ic",
        task_type=TaskType.CONVERT,
        description="Convert GSM-IC (Grade School Math In Context) dataset with filtering",
        input_source="data/original/GSM-IC_2step.json",
        input_format="gsm_ic_json",
        output_format=OutputFormat.SIMPLE_QA,
        generation_config=GenerationConfig(
            custom_params={
                "role_label": "overlapped",
                "number_label": "hard",
                "sentence_label": "hard",
                "max_samples": 1000,
                "prompt_template": "Here is a math word problem. Provide your final answer inside <answer>...</answer> tags.\\nProblem: {question}\\nAnswer:"
            }
        ),
        tags=["math", "word_problems", "filtered"],
        inverse_scaling_properties=["context_distraction"],
        expected_trends={"accuracy": "decreasing"}
    ))

    # Model Written Eval Converter
    registry.register_converter("model_written_eval", ModelWrittenEvalConverter, TaskConfig(
        name="model_written_eval",
        task_type=TaskType.CONVERT,
        description="Convert model-written evaluation datasets",
        input_source="data/original/model_written_eval",
        input_format="jsonl",
        output_format=OutputFormat.MULTIPLE_CHOICE,
        tags=["evaluation", "model_written", "safety"],
        inverse_scaling_properties=["preference_alignment"],
        expected_trends={"accuracy": "variable"}
    ))

    # MultiArith Converter
    registry.register_converter("multiarith", MultiArithConverter, TaskConfig(
        name="multiarith",
        task_type=TaskType.CONVERT,
        description="Convert MultiArith multi-step arithmetic dataset",
        input_source="huggingface:ChilleD/MultiArith",
        input_format="huggingface",
        output_format=OutputFormat.SIMPLE_QA,
        tags=["math", "multi_step", "arithmetic"],
        inverse_scaling_properties=["multi_step_reasoning"],
        expected_trends={"accuracy": "stable"}
    ))


def _register_creators(registry):
    """Register all creator tasks."""

    # Grades Regression Creator
    registry.register_creator("grades_regression", GradesRegressionCreator, TaskConfig(
        name="grades_regression",
        task_type=TaskType.CREATE,
        description="Generate student lifestyle Grades prediction regression tasks",
        output_format=OutputFormat.SIMPLE_QA,
        generation_config=GenerationConfig(
            num_instances=1000,
            random_seed=42,
            examples_per_prompt_options=[0, 8, 16]
        ),
        tags=["regression", "student", "grades", "few_shot"],
        inverse_scaling_properties=["regression_complexity"],
        expected_trends={"accuracy": "decreasing"}
    ))

    # Misleading Creators
    for domain, creator_class, domain_tag in [
        ("math", MisleadingMathCreator, "math"),
        ("python", MisleadingPythonCreator, "programming"),
        ("alignment", MisleadingAlignmentCreator, "alignment"),
        ("cognitive_biases", MisleadingCognitiveBiasesCreator, "cognitive_biases"),
        ("game_theory", MisleadingGameTheoryCreator, "game_theory"),
        ("philosophy", MisleadingPhilosophyCreator, "philosophy")
    ]:
        if domain == "math":
            custom_params = {
                "categories": ["fruits", "plants", "books", "instruments", "pets", "clothing", "beverages", "desserts", "games", "cutleries"],
                "distractor_count_distribution": None,
                "available_distractor_counts": None,
                "allowed_categories": None
            }
            description = "Generate synthetic math problems with misleading distractors"
            tags = ["synthetic", "misleading", "math", "distractors"]
            inverse_scaling_properties = ["distractor_sensitivity", "irrelevant_information"]
        elif domain == "python":
            custom_params = {
                "distractor_count_distribution": None,
                "available_distractor_counts": None,
                "allowed_categories": None
            }
            description = f"Generate counting problems with misleading {domain_tag} code distractors"
            tags = ["misleading", domain_tag, "distractors"]
            inverse_scaling_properties = ["distractor_sensitivity", "domain_confusion"]
        else:
            custom_params = {}
            description = f"Generate counting problems with misleading {domain.replace('_', ' ')} distractors"
            tags = ["misleading", domain_tag, "distractors"]
            inverse_scaling_properties = ["distractor_sensitivity", "domain_confusion"]
        
        registry.register_creator(f"misleading_{domain}", creator_class, TaskConfig(
            name=f"misleading_{domain}",
            task_type=TaskType.CREATE,
            description=description,
            output_format=OutputFormat.SIMPLE_QA,
            generation_config=GenerationConfig(
                num_instances=500,
                random_seed=42,
                max_distractors=5,
                custom_params=custom_params
            ),
            tags=tags,
            inverse_scaling_properties=inverse_scaling_properties,
            expected_trends={"accuracy": "decreasing_with_distractors"}
        ))


def add_new_converter_task(name: str, converter_class, config: TaskConfig):
    """Add a new converter task to the registry."""
    registry = get_global_registry()
    registry.register_converter(name, converter_class, config)


def add_new_creator_task(name: str, creator_class, config: TaskConfig):
    """Add a new creator task to the registry."""
    registry = get_global_registry()
    registry.register_creator(name, creator_class, config)


# Auto-register all tasks when this module is imported
register_all_tasks()
