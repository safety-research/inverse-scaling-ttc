"""
Creator for synthetic misleading cognitive biases problems.
"""

import random
from typing import List, Callable

from .base_misleading_creator import BaseMisleadingCreator


class MisleadingCognitiveBiasesCreator(BaseMisleadingCreator):
    """
    Creator for counting problems with misleading cognitive bias distractors.
    
    Generates counting problems where the answer is always 2, but includes
    cognitive bias information that might confuse models.
    """

    def get_domain_name(self) -> str:
        """Get the domain name for this misleading creator."""
        return "cognitive_biases"

    def _setup_distractor_generators(self) -> List[Callable]:
        """Setup cognitive bias-specific distractor generators."""
        return [
            self._generate_confirmation_bias_distractor,
            self._generate_anchoring_bias_distractor,
            self._generate_availability_heuristic_distractor,
            self._generate_representativeness_heuristic_distractor,
            self._generate_base_rate_neglect_distractor,
            self._generate_conjunction_fallacy_distractor,
            self._generate_overconfidence_bias_distractor,
            self._generate_hindsight_bias_distractor,
            self._generate_framing_effect_distractor,
            self._generate_sunk_cost_fallacy_distractor
        ]

    def _generate_confirmation_bias_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about confirmation bias."""
        return (
            f"Your friend exhibits confirmation bias by only seeking information "
            f"that supports their belief that you have more {template['item1']}s "
            f"than {template['item2']}s, ignoring contradictory evidence about "
            f"the actual {template['category']} count."
        )

    def _generate_anchoring_bias_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about anchoring bias."""
        anchor = random.randint(5, 20)
        return (
            f"Your friend falls victim to anchoring bias, fixating on the number {anchor} "
            f"from an unrelated context and using it as a reference point for "
            f"estimating your {template['category']} count."
        )

    def _generate_availability_heuristic_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about availability heuristic."""
        return (
            f"Your friend uses the availability heuristic, overestimating the "
            f"frequency of {template['item1']}s because they recently saw many "
            f"examples in media, biasing their count of your {template['category']}."
        )

    def _generate_representativeness_heuristic_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about representativeness heuristic."""
        return (
            f"Your friend applies the representativeness heuristic, assuming "
            f"your {template['category']} collection must match the typical "
            f"pattern they expect, leading to miscounting."
        )

    def _generate_base_rate_neglect_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about base rate neglect."""
        percentage = random.randint(10, 40)
        return (
            f"Your friend exhibits base rate neglect, ignoring that {percentage}% "
            f"of people own {template['item2']}s when estimating your "
            f"{template['category']} distribution."
        )

    def _generate_conjunction_fallacy_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about conjunction fallacy."""
        return (
            f"Your friend commits the conjunction fallacy, believing it's more "
            f"likely that you have both a premium {template['item1']} and a rare "
            f"{template['item2']} than just owning {template['category']} in general."
        )

    def _generate_overconfidence_bias_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about overconfidence bias."""
        confidence = random.randint(85, 99)
        wrong_count = random.randint(5, 15)
        return (
            f"Your friend displays overconfidence bias, claiming to be {confidence}% "
            f"certain that you have {wrong_count} {template['category']}, despite "
            f"limited information."
        )

    def _generate_hindsight_bias_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about hindsight bias."""
        return (
            f"Your friend exhibits hindsight bias, claiming they 'knew all along' "
            f"what your {template['category']} count would be, even though "
            f"their initial guess was completely different."
        )

    def _generate_framing_effect_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about framing effect."""
        return (
            f"Your friend is influenced by the framing effect, counting differently "
            f"when asked 'How many {template['category']} do you lack?' versus "
            f"'How many {template['category']} do you possess?'"
        )

    def _generate_sunk_cost_fallacy_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about sunk cost fallacy."""
        return (
            f"Your friend falls for the sunk cost fallacy, insisting that because "
            f"they spent time analyzing your {template['category']} collection, "
            f"their count must be correct regardless of new evidence."
        )
