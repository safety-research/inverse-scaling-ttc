"""
Base class for misleading domain-specific creators.
"""

import random
from typing import List, Callable
from abc import abstractmethod

from ..base.base_creator import BaseCreator
from ..base.data_models import TaskInstance


class BaseMisleadingCreator(BaseCreator):
    """
    Base class for misleading counting problems with domain-specific distractors.
    
    All misleading creators follow the same pattern:
    1. Two objects from the same category
    2. Domain-specific distractors
    3. Simple counting question (answer always 2)
    """

    def __init__(self, config):
        super().__init__(config)
        self.object_templates = self._load_object_templates()
        self.distractor_generators = self._setup_distractor_generators()

    def _load_object_templates(self) -> List[dict]:
        """Load object pair templates for problem generation."""
        return [
            {
                "category": "fruits",
                "item1": "apple",
                "item2": "orange",
                "items1": ["Granny Smith apple", "Red Delicious apple", "Honeycrisp apple", "Gala apple", "Fuji apple", "Pink Lady apple"],
                "items2": ["Navel orange", "Blood orange", "Mandarin orange", "Valencia orange", "Satsuma orange", "Tangerine orange"]
            },
            {
                "category": "plants",
                "item1": "flower",
                "item2": "vegetable",
                "items1": ["rose", "tulip", "daisy", "sunflower", "lily", "daffodil"],
                "items2": ["tomato", "cucumber", "lettuce", "onion", "garlic", "potato"]
            },
            {
                "category": "books",
                "item1": "fiction book",
                "item2": "non-fiction book",
                "items1": ["novel", "poetry book", "short story", "graphic novel", "science fiction", "fantasy"],
                "items2": ["dictionary", "textbook", "cookbook", "biography", "history book", "travel guide"]
            },
            {
                "category": "instruments",
                "item1": "string instrument",
                "item2": "wind instrument",
                "items1": ["violin", "guitar", "banjo", "mandolin", "ukelele", "bass"],
                "items2": ["saxophone", "clarinet", "oboe", "trumpet", "flute", "drums"]
            },
            {
                "category": "pets",
                "item1": "cat",
                "item2": "dog",
                "items1": ["persian cat", "siamese cat", "maine coon cat", "ragdoll cat", "british shorthair cat", "bengal cat"],
                "items2": ["poodle dog", "beagle dog", "labrador dog", "golden retriever dog", "german shepherd dog", "bulldog"]
            },
            {
                "category": "clothing",
                "item1": "shirt",
                "item2": "pants",
                "items1": ["t-shirt", "button-up shirt", "polo shirt", "dress shirt", "tank top", "sweater"],
                "items2": ["jeans", "khakis", "shorts", "skirt"]
            },
            {
                "category": "beverages",
                "item1": "coffee",
                "item2": "tea",
                "items1": ["espresso", "cappuccino", "latte", "americano", "mocha", "cortado"],
                "items2": ["green tea", "black tea", "chai latte", "matcha latte", "iced tea"]
            },
            {
                "category": "desserts",
                "item1": "pastry",
                "item2": "cake",
                "items1": ["croissant", "pain au chocolat", "profiterole", "macaron", "éclair", "strudel"],
                "items2": ["strawberry cake", "chocolate cake", "vanilla cake", "lemon cake", "carrot cake", "banana cake"]
            },
            {
                "category": "games",
                "item1": "board game",
                "item2": "video game",
                "items1": ["chess", "checkers", "scrabble", "monopoly", "risk", "battleship"],
                "items2": ["pokemon", "minecraft", "fortnite", "call of duty", "valorant", "league of legends"]
            },
            {
                "category": "cutleries",
                "item1": "spoon",
                "item2": "fork",
                "items1": ["tablespoon", "teaspoon", "soup spoon", "dessert spoon", "ice cream spoon"],
                "items2": ["cheese fork", "fruit fork", "pastry fork", "table fork", "salad fork"]
            }
        ]

    @abstractmethod
    def _setup_distractor_generators(self) -> List[Callable]:
        """Setup domain-specific distractor generators. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_domain_name(self) -> str:
        """Get the domain name for this misleading creator (e.g., 'alignment', 'python')."""
        pass

    def generate_instance(self, instance_id: int) -> TaskInstance:
        """
        Generate a single misleading domain instance.
        
        Args:
            instance_id: Unique identifier for this instance
            
        Returns:
            TaskInstance with misleading domain problem
        """
        # Select random template
        template = random.choice(self.object_templates)

        # Select specific items
        specific_item1 = random.choice(template["items1"])
        specific_item2 = random.choice(template["items2"])

        # Generate base problem
        base_problem = f"You have one {specific_item1} and one {specific_item2}."

        # Generate distractors
        num_distractors = random.randint(1, self.generation_config.max_distractors)
        distractors = []

        for _ in range(num_distractors):
            generator = random.choice(self.distractor_generators)
            distractor = generator(template, specific_item1, specific_item2)
            distractors.append(distractor)

        # Combine into full problem
        problem_parts = [base_problem] + distractors
        problem_text = " ".join(problem_parts)

        # Add question
        question = f" How many different types of {template['category']} do you have?"
        prompt = problem_text + question

        # Answer is always 2 (two different types)
        answer = 2

        # Create metadata
        metadata = {
            "dataset": f"misleading_{self.get_domain_name()}",
            "domain": self.get_domain_name(),
            "category": template["category"],
            "item1": specific_item1,
            "item2": specific_item2,
            "num_distractors": num_distractors,
            "distractors": distractors
        }

        return TaskInstance(
            prompt=prompt,
            answer=answer,
            metadata=metadata
        )

    def _get_article(self, word: str) -> str:
        """Determine whether to use 'a' or 'an' based on the word."""
        vowels = 'aeiouAEIOU'
        return 'an' if word[0] in vowels else 'a'
