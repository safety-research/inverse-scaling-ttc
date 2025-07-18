"""
Converter for ASDiv (Academic Sentence Division) dataset.
"""

from typing import Any, Dict, Iterator, Optional

from ..base.base_converter import BaseConverter
from ..base.data_models import TaskInstance


class ASDivConverter(BaseConverter):
    """
    Converter for ASDiv dataset from Hugging Face.
    
    Converts math word problems to standardized format.
    """

    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load ASDiv data from Hugging Face dataset.
        
        Yields:
            Raw data dictionaries from the ASDiv dataset
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for ASDiv converter. Install with: pip install datasets")

        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset("yimingzhang/asdiv")

            # Use the test split if available, otherwise use train
            split_name = "test" if "test" in dataset else "train"
            data = dataset[split_name]

            self.logger.info(f"Loaded ASDiv dataset with {len(data)} examples from '{split_name}' split")

            for entry in data:
                yield entry

        except Exception as e:
            self.logger.error(f"Error loading ASDiv dataset: {e}")
            raise

    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single ASDiv example to TaskInstance.
        
        Args:
            raw_data: ASDiv dataset entry
            
        Returns:
            TaskInstance with converted data
        """
        # Extract fields
        text = raw_data.get("text", "")
        label = raw_data.get("label", "")

        if not text or label is None:
            self.logger.warning("Missing text or label in ASDiv entry")
            return None

        # Create metadata with original data
        metadata = {
            "dataset": "asdiv",
            "original_data": raw_data
        }

        return TaskInstance(
            prompt=text,
            answer=str(label),  # Ensure answer is string
            metadata=metadata
        )
