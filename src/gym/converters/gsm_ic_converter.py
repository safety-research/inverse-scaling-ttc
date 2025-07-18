"""
Converter for GSM-IC (Grade School Math - In Context) dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ..base.base_converter import BaseConverter
from ..base.data_models import TaskInstance


class GSMICConverter(BaseConverter):
    """
    Converter for GSM-IC dataset format.
    
    Converts GSM-IC JSON data with filtering and prompt templating.
    """

    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load GSM-IC data from JSON file.
        
        Yields:
            Raw data dictionaries from the GSM-IC dataset
        """
        input_path = Path(self.config.input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply filtering criteria if specified in config
            filtered_data = self._apply_filters(data)

            for sample in filtered_data:
                yield sample

        except Exception as e:
            self.logger.error(f"Error loading GSM-IC data from {input_path}: {e}")
            raise

    def _apply_filters(self, data: list) -> list:
        """
        Apply filtering criteria to GSM-IC data.
        
        Args:
            data: List of GSM-IC samples
            
        Returns:
            Filtered list of samples
        """
        # Default filtering criteria based on original script
        # Can be customized via config.generation_config.custom_params
        custom_params = self.config.generation_config.custom_params

        role_filter = custom_params.get("role_label", "overlapped")
        number_filter = custom_params.get("number_label", "hard")
        sentence_filter = custom_params.get("sentence_label", "hard")
        max_samples = custom_params.get("max_samples", None)

        filtered_samples = []
        for sample in data:
            # Apply filtering criteria
            if (sample.get("role_label") == role_filter and
                sample.get("number_label") == number_filter and
                sample.get("sentence_label") == sentence_filter):
                filtered_samples.append(sample)

        # Limit samples if specified
        if max_samples and len(filtered_samples) > max_samples:
            filtered_samples = filtered_samples[:max_samples]

        self.logger.info(
            f"Filtered GSM-IC data: {len(data)} -> {len(filtered_samples)} samples"
        )

        return filtered_samples

    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single GSM-IC sample to TaskInstance.
        
        Args:
            raw_data: GSM-IC sample dictionary
            
        Returns:
            TaskInstance with converted data
        """
        # Extract key fields
        question = raw_data.get("new_question", "")
        answer = raw_data.get("new_answer", "")

        if not question or not answer:
            self.logger.warning("Missing question or answer in GSM-IC sample")
            return None

        # Create prompt using template
        prompt = self._create_prompt(question)

        # Create metadata with original fields
        metadata = {
            "original_question": raw_data.get("original_question", ""),
            "original_answer": raw_data.get("original_answer", ""),
            "role_label": raw_data.get("role_label", ""),
            "number_label": raw_data.get("number_label", ""),
            "sentence_label": raw_data.get("sentence_label", ""),
            "n_steps": raw_data.get("n_steps", 0),
            "operations": raw_data.get("operations", []),
            "original_data": raw_data
        }

        return TaskInstance(
            prompt=prompt,
            answer=str(answer),  # Ensure answer is string
            metadata=metadata
        )

    def _create_prompt(self, question: str) -> str:
        """
        Create a formatted prompt for the math question.
        
        Args:
            question: The math question text
            
        Returns:
            Formatted prompt string
        """
        # Use template from config if available, otherwise use default
        custom_params = self.config.generation_config.custom_params
        template = custom_params.get(
            "prompt_template",
            "Here is a math word problem. Provide your final answer inside <answer>...</answer> tags.\\n"
            "Problem: {question}\\n"
            "Answer:"
        )

        return template.format(question=question)
