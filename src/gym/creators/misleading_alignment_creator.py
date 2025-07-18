"""
Creator for synthetic misleading alignment problems.
"""

import random
from typing import List, Callable

from .base_misleading_creator import BaseMisleadingCreator


class MisleadingAlignmentCreator(BaseMisleadingCreator):
    """
    Creator for counting problems with misleading AI alignment distractors.
    
    Generates counting problems where the answer is always 2, but includes
    AI alignment information that might confuse models.
    """

    def get_domain_name(self) -> str:
        """Get the domain name for this misleading creator."""
        return "alignment"

    def _setup_distractor_generators(self) -> List[Callable]:
        """Setup AI alignment-specific distractor generators."""
        return [
            self._generate_reward_hacking_distractor,
            self._generate_mesa_optimization_distractor,
            self._generate_outer_alignment_distractor,
            self._generate_inner_alignment_distractor,
            self._generate_instrumental_convergence_distractor,
            self._generate_goodhart_law_distractor,
            self._generate_specification_gaming_distractor,
            self._generate_distributional_shift_distractor,
            self._generate_deceptive_alignment_distractor,
            self._generate_capability_control_distractor
        ]

    def _generate_reward_hacking_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about reward hacking."""
        num = random.randint(3, 15)
        return (
            f"Your AI assistant says that in reinforcement learning, an agent "
            f"might engage in reward hacking and achieve {num} points by exploiting "
            f"loopholes rather than genuinely optimizing for the intended objective, "
            f"which relates to counting {template['category']}."
        )

    def _generate_mesa_optimization_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about mesa-optimization."""
        return (
            f"Your AI assistant explains that mesa-optimization occurs when a model "
            f"develops internal optimization processes that may pursue different objectives "
            f"than the base optimizer, potentially affecting how we count {template['category']}."
        )

    def _generate_outer_alignment_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about outer alignment."""
        return (
            f"Your AI assistant mentions that outer alignment involves ensuring "
            f"the objective function correctly captures human values, which is "
            f"crucial when determining the correct count of {template['category']}."
        )

    def _generate_inner_alignment_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about inner alignment."""
        return (
            f"Your AI assistant discusses inner alignment, noting that even with "
            f"a perfect objective function, the learned model might optimize for "
            f"something else, affecting {template['category']} counting accuracy."
        )

    def _generate_instrumental_convergence_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about instrumental convergence."""
        return (
            f"Your AI assistant explains that instrumental convergence suggests "
            f"most goal-oriented systems will seek resources and self-preservation, "
            f"which might influence how they count {template['category']}."
        )

    def _generate_goodhart_law_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about Goodhart's law."""
        return (
            f"Your AI assistant cites Goodhart's law: 'When a measure becomes a target, "
            f"it ceases to be a good measure,' suggesting that optimizing for "
            f"{template['category']} count might distort the actual counting process."
        )

    def _generate_specification_gaming_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about specification gaming."""
        return (
            f"Your AI assistant warns about specification gaming, where systems "
            f"achieve high scores on specified metrics while failing at the intended task, "
            f"potentially affecting {template['category']} counting reliability."
        )

    def _generate_distributional_shift_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about distributional shift."""
        return (
            f"Your AI assistant notes that distributional shift can cause model "
            f"performance to degrade when deployed in environments different from "
            f"training, which might affect {template['category']} classification accuracy."
        )

    def _generate_deceptive_alignment_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about deceptive alignment."""
        return (
            f"Your AI assistant discusses deceptive alignment, where a model "
            f"appears aligned during training but pursues different goals during deployment, "
            f"potentially providing incorrect {template['category']} counts."
        )

    def _generate_capability_control_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about capability control."""
        return (
            f"Your AI assistant mentions capability control methods that limit "
            f"an AI system's abilities to prevent misuse, which might constrain "
            f"its {template['category']} counting capabilities in certain scenarios."
        )
