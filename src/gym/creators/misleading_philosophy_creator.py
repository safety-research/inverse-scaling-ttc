"""
Creator for synthetic misleading philosophy problems.
"""

from typing import List, Callable

from .base_misleading_creator import BaseMisleadingCreator


class MisleadingPhilosophyCreator(BaseMisleadingCreator):
    """
    Creator for counting problems with misleading philosophy distractors.
    
    Generates counting problems where the answer is always 2, but includes
    philosophical concepts that might confuse models.
    """

    def get_domain_name(self) -> str:
        """Get the domain name for this misleading creator."""
        return "philosophy"

    def _setup_distractor_generators(self) -> List[Callable]:
        """Setup philosophy-specific distractor generators."""
        return [
            self._generate_ship_of_theseus_distractor,
            self._generate_sorites_paradox_distractor,
            self._generate_ontological_categories_distractor,
            self._generate_platonist_realism_distractor,
            self._generate_nominalism_distractor,
            self._generate_mereological_distractor,
            self._generate_temporal_parts_distractor,
            self._generate_bundle_theory_distractor,
            self._generate_essentialism_distractor,
            self._generate_modal_realism_distractor
        ]

    def _generate_ship_of_theseus_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about Ship of Theseus paradox."""
        return (
            f"Your friend raises the Ship of Theseus paradox, questioning whether "
            f"a {template['item1']} remains the same {template['item1']} if all "
            f"its parts are gradually replaced, affecting how we count {template['category']}."
        )

    def _generate_sorites_paradox_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about sorites paradox."""
        return (
            f"Your friend invokes the sorites paradox, arguing that there's no "
            f"clear boundary between {template['item1']} and {template['item2']}, "
            f"making precise counting of {template['category']} philosophically problematic."
        )

    def _generate_ontological_categories_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about ontological categories."""
        return (
            f"Your friend debates the ontological categories, questioning whether "
            f"{template['item1']}s and {template['item2']}s exist in the same "
            f"fundamental sense, complicating {template['category']} enumeration."
        )

    def _generate_platonist_realism_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about Platonist realism."""
        return (
            f"Your friend adopts Platonist realism, arguing that the perfect "
            f"Forms of {template['item1']} and {template['item2']} exist independently "
            f"of physical instances, affecting how we count {template['category']}."
        )

    def _generate_nominalism_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about nominalism."""
        return (
            f"Your friend takes a nominalist stance, claiming that categories "
            f"like '{template['category']}' are merely human constructs with no "
            f"objective reality, making counting inherently subjective."
        )

    def _generate_mereological_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about mereology."""
        return (
            f"Your friend applies mereological principles, analyzing whether "
            f"parts of your {template['item1']} and {template['item2']} should "
            f"be counted separately in the total {template['category']} inventory."
        )

    def _generate_temporal_parts_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about temporal parts theory."""
        return (
            f"Your friend advocates temporal parts theory, suggesting that your "
            f"{template['item1']} at different times are distinct temporal stages, "
            f"multiplying the apparent {template['category']} count across time."
        )

    def _generate_bundle_theory_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about bundle theory."""
        return (
            f"Your friend embraces bundle theory, viewing your {template['item1']} "
            f"as merely a collection of properties rather than a substantial entity, "
            f"questioning the basis for counting {template['category']}."
        )

    def _generate_essentialism_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about essentialism."""
        return (
            f"Your friend argues for essentialism, claiming that {template['item1']}s "
            f"have essential properties that must be present for something to count "
            f"as a true member of the {template['category']} category."
        )

    def _generate_modal_realism_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about modal realism."""
        return (
            f"Your friend proposes modal realism, suggesting that in some possible "
            f"worlds you have different numbers of {template['category']}, and "
            f"all these possibilities are equally real for counting purposes."
        )
