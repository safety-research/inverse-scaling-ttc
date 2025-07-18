"""
Task registry for managing available tasks and their configurations.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from .base_converter import BaseConverter
from .base_creator import BaseCreator
from .data_models import TaskConfig, TaskType

logger = logging.getLogger(__name__)


class TaskRegistry:
    """
    Registry for managing available tasks and their implementations.
    
    This class maintains a registry of all available converter and creator tasks,
    allowing for dynamic discovery and instantiation of task implementations.
    """

    def __init__(self):
        """Initialize an empty task registry."""
        self._converters: Dict[str, Type[BaseConverter]] = {}
        self._creators: Dict[str, Type[BaseCreator]] = {}
        self._configs: Dict[str, TaskConfig] = {}

    def register_converter(self, name: str, converter_class: Type[BaseConverter],
                          config: Optional[TaskConfig] = None) -> None:
        """
        Register a converter task.
        
        Args:
            name: Unique name for the converter
            converter_class: Class implementing BaseConverter
            config: Optional default configuration
        """
        if not issubclass(converter_class, BaseConverter):
            raise ValueError("Converter class must inherit from BaseConverter")

        self._converters[name] = converter_class

        if config is not None:
            if config.task_type != TaskType.CONVERT:
                raise ValueError("Config for converter must have task_type=CONVERT")
            self._configs[name] = config

        logger.debug(f"Registered converter: {name}")

    def register_creator(self, name: str, creator_class: Type[BaseCreator],
                        config: Optional[TaskConfig] = None) -> None:
        """
        Register a creator task.
        
        Args:
            name: Unique name for the creator
            creator_class: Class implementing BaseCreator
            config: Optional default configuration
        """
        if not issubclass(creator_class, BaseCreator):
            raise ValueError("Creator class must inherit from BaseCreator")

        self._creators[name] = creator_class

        if config is not None:
            if config.task_type != TaskType.CREATE:
                raise ValueError("Config for creator must have task_type=CREATE")
            self._configs[name] = config

        logger.debug(f"Registered creator: {name}")

    def get_converter(self, name: str) -> Type[BaseConverter]:
        """
        Get a registered converter class.
        
        Args:
            name: Name of the converter
            
        Returns:
            Converter class
            
        Raises:
            KeyError: If converter not found
        """
        if name not in self._converters:
            raise KeyError(f"Converter '{name}' not found. Available: {list(self._converters.keys())}")
        return self._converters[name]

    def get_creator(self, name: str) -> Type[BaseCreator]:
        """
        Get a registered creator class.
        
        Args:
            name: Name of the creator
            
        Returns:
            Creator class
            
        Raises:
            KeyError: If creator not found
        """
        if name not in self._creators:
            raise KeyError(f"Creator '{name}' not found. Available: {list(self._creators.keys())}")
        return self._creators[name]

    def get_config(self, name: str) -> Optional[TaskConfig]:
        """
        Get the default configuration for a task.
        
        Args:
            name: Name of the task
            
        Returns:
            TaskConfig if available, None otherwise
        """
        return self._configs.get(name)

    def list_converters(self) -> List[str]:
        """List all registered converter names."""
        return list(self._converters.keys())

    def list_creators(self) -> List[str]:
        """List all registered creator names."""
        return list(self._creators.keys())

    def list_all_tasks(self) -> List[str]:
        """List all registered task names (both converters and creators)."""
        return self.list_converters() + self.list_creators()

    def create_task(self, name: str, config: Optional[TaskConfig] = None) -> Union[BaseConverter, BaseCreator]:
        """
        Create an instance of a registered task.
        
        Args:
            name: Name of the task
            config: Optional configuration (uses default if not provided)
            
        Returns:
            Instance of the task (converter or creator)
            
        Raises:
            KeyError: If task not found
            ValueError: If configuration is invalid
        """
        # Determine if it's a converter or creator
        if name in self._converters:
            task_class = self._converters[name]
            task_type = TaskType.CONVERT
        elif name in self._creators:
            task_class = self._creators[name]
            task_type = TaskType.CREATE
        else:
            raise KeyError(f"Task '{name}' not found")

        # Use provided config or default
        if config is None:
            config = self.get_config(name)
            if config is None:
                raise ValueError(f"No configuration provided for task '{name}' and no default available")

        # Validate config type matches task type
        if config.task_type != task_type:
            raise ValueError(f"Config task_type {config.task_type} doesn't match task type {task_type}")

        return task_class(config)

    def auto_discover_tasks(self, search_paths: List[Union[str, Path]]) -> int:
        """
        Automatically discover and register tasks from specified paths.
        
        Args:
            search_paths: List of paths to search for task implementations
            
        Returns:
            Number of tasks discovered and registered
        """
        discovered_count = 0

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                logger.warning(f"Search path does not exist: {path}")
                continue

            # Look for Python files that might contain task implementations
            for py_file in path.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                try:
                    # Try to import and inspect the module
                    module_name = str(py_file.relative_to(path.parent)).replace("/", ".").replace("\\", ".")[:-3]
                    module = importlib.import_module(module_name)

                    # Look for classes that inherit from BaseConverter or BaseCreator
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and
                            attr not in (BaseConverter, BaseCreator)):

                            if issubclass(attr, BaseConverter):
                                task_name = py_file.stem
                                self.register_converter(task_name, attr)
                                discovered_count += 1

                            elif issubclass(attr, BaseCreator):
                                task_name = py_file.stem
                                self.register_creator(task_name, attr)
                                discovered_count += 1

                except Exception as e:
                    logger.debug(f"Could not import {py_file}: {e}")

        logger.info(f"Auto-discovered {discovered_count} tasks from {len(search_paths)} search paths")
        return discovered_count


# Global registry instance
_global_registry = TaskRegistry()


def get_global_registry() -> TaskRegistry:
    """Get the global task registry instance."""
    return _global_registry
