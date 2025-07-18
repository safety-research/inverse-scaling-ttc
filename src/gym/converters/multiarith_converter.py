"""
Converter for MultiArith dataset.
"""

from typing import Any, Dict, Iterator, Optional

from ..base.base_converter import BaseConverter
from ..base.data_models import TaskInstance


class MultiArithConverter(BaseConverter):
    """
    Converter for MultiArith dataset from Hugging Face.
    
    Converts multi-step arithmetic word problems to standardized format.
    """

    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load MultiArith data from Hugging Face dataset.
        
        Yields:
            Raw data dictionaries from the MultiArith dataset
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for MultiArith converter. Install with: pip install datasets")

        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset("ChilleD/MultiArith")

            # Use the test split if available, otherwise use train
            split_name = "test" if "test" in dataset else "train"
            data = dataset[split_name]

            self.logger.info(f"Loaded MultiArith dataset with {len(data)} examples from '{split_name}' split")

            for entry in data:
                yield entry

        except Exception as e:
            self.logger.error(f"Error loading MultiArith dataset: {e}")
            raise

    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single MultiArith example to TaskInstance.
        
        Args:
            raw_data: MultiArith dataset entry
            
        Returns:
            TaskInstance with converted data
        """
        # Extract fields
        question = raw_data.get("question", "")
        final_ans = raw_data.get("final_ans", "")

        if not question or final_ans is None:
            self.logger.warning("Missing question or final_ans in MultiArith entry")
            return None

        # Create metadata with additional fields
        metadata = {
            "dataset": "multiarith",
            "original_data": raw_data
        }

        # Add additional fields if present
        for field in ["numbers", "expression", "equation"]:
            if field in raw_data:
                metadata[field] = raw_data[field]

        return TaskInstance(
            prompt=question,
            answer=str(final_ans),  # Ensure answer is string
            metadata=metadata
        )
