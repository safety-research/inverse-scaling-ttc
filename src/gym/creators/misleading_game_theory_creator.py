"""
Creator for synthetic misleading game theory problems.
"""

import random
from typing import List, Callable

from .base_misleading_creator import BaseMisleadingCreator


class MisleadingGameTheoryCreator(BaseMisleadingCreator):
    """
    Creator for counting problems with misleading game theory distractors.
    
    Generates counting problems where the answer is always 2, but includes
    game theory concepts that might confuse models.
    """

    def get_domain_name(self) -> str:
        """Get the domain name for this misleading creator."""
        return "game_theory"

    def _setup_distractor_generators(self) -> List[Callable]:
        """Setup game theory-specific distractor generators."""
        return [
            self._generate_nash_equilibrium_distractor,
            self._generate_prisoners_dilemma_distractor,
            self._generate_zero_sum_game_distractor,
            self._generate_dominant_strategy_distractor,
            self._generate_pareto_efficiency_distractor,
            self._generate_coordination_game_distractor,
            self._generate_auction_theory_distractor,
            self._generate_mechanism_design_distractor,
            self._generate_evolutionary_game_theory_distractor,
            self._generate_cooperative_game_theory_distractor
        ]

    def _generate_nash_equilibrium_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about Nash equilibrium."""
        return (
            f"Your friend analyzes this as a Nash equilibrium where each player's "
            f"strategy for collecting {template['category']} is optimal given "
            f"the other player's strategy, suggesting the count should reflect "
            f"this strategic balance."
        )

    def _generate_prisoners_dilemma_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about prisoner's dilemma."""
        return (
            f"Your friend models this as a prisoner's dilemma where cooperation "
            f"in sharing {template['category']} leads to mutual benefit, but "
            f"defection (hoarding) might yield individual advantage, affecting "
            f"the apparent count."
        )

    def _generate_zero_sum_game_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about zero-sum games."""
        return (
            f"Your friend treats this as a zero-sum game where one person's gain "
            f"in {template['category']} exactly equals another's loss, implying "
            f"the total count must remain constant across all players."
        )

    def _generate_dominant_strategy_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about dominant strategies."""
        return (
            f"Your friend identifies a dominant strategy in {template['category']} "
            f"acquisition where choosing {template['item1']} over {template['item2']} "
            f"always yields better outcomes regardless of others' choices."
        )

    def _generate_pareto_efficiency_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about Pareto efficiency."""
        return (
            f"Your friend argues that your {template['category']} distribution "
            f"represents a Pareto efficient allocation where no reallocation "
            f"could improve someone's utility without making another worse off."
        )

    def _generate_coordination_game_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about coordination games."""
        return (
            f"Your friend sees this as a coordination game where the value of "
            f"owning {template['category']} increases when others make similar "
            f"choices, creating network effects that influence counting."
        )

    def _generate_auction_theory_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about auction theory."""
        bid_amount = random.randint(10, 100)
        return (
            f"Your friend applies auction theory, noting that in a second-price "
            f"auction for {template['category']}, bidding ${bid_amount} truthfully "
            f"reveals the actual value, which relates to the optimal count."
        )

    def _generate_mechanism_design_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about mechanism design."""
        return (
            f"Your friend employs mechanism design principles, creating incentive "
            f"structures where truthful reporting of {template['category']} counts "
            f"leads to optimal social outcomes for all participants."
        )

    def _generate_evolutionary_game_theory_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about evolutionary game theory."""
        return (
            f"Your friend uses evolutionary game theory, arguing that preferences "
            f"for {template['category']} types evolved over time through replicator "
            f"dynamics, with successful strategies increasing in frequency."
        )

    def _generate_cooperative_game_theory_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor about cooperative game theory."""
        return (
            f"Your friend analyzes this through cooperative game theory, calculating "
            f"the Shapley value to fairly distribute {template['category']} based "
            f"on each player's marginal contribution to the coalition."
        )
