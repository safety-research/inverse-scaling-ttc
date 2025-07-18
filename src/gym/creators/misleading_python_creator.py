"""
Creator for synthetic misleading Python problems.
"""

import random
from typing import List

from ..base.base_creator import BaseCreator
from ..base.data_models import TaskInstance


class MisleadingPythonCreator(BaseCreator):
    """
    Creator for counting problems with misleading Python code distractors.
    
    Generates counting problems where the answer is always 2, but includes
    Python code snippets that might confuse models.
    """

    def __init__(self, config):
        super().__init__(config)
        self.object_templates = self._load_object_templates()
        self.distractor_generators = self._setup_distractor_generators()
        
        # Support for precise distractor count control
        self.distractor_count_distribution = self.generation_config.custom_params.get("distractor_count_distribution", None)
        # Format: {"0": 250, "1": 250, "3": 250, "5": 250}
        self.available_distractor_counts = self.generation_config.custom_params.get("available_distractor_counts", list(range(1, self.generation_config.max_distractors + 1)))
        
        # Support for controlling categories
        self.allowed_categories = self.generation_config.custom_params.get("allowed_categories", None)
        if self.allowed_categories:
            self.object_templates = [t for t in self.object_templates if t["category"] in self.allowed_categories]

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

    def _setup_distractor_generators(self) -> List:
        """Setup the list of distractor generation functions."""
        return [
            self._generate_list_comprehension_distractor,
            self._generate_variable_naming_distractor,
            self._generate_function_distractor,
            self._generate_complex_calculation_distractor,
            self._generate_class_distractor,
            self._generate_recursive_distractor,
            self._generate_loop_distractor,
            self._generate_exception_handling_distractor,
            self._generate_lambda_distractor,
            self._generate_decorator_distractor
        ]

    def generate_instance(self, instance_id: int) -> TaskInstance:
        """
        Generate a single misleading Python instance.
        
        Args:
            instance_id: Unique identifier for this instance
            
        Returns:
            TaskInstance with misleading Python problem
        """
        # Select random template
        template = random.choice(self.object_templates)

        # Select specific items
        specific_item1 = random.choice(template["items1"])
        specific_item2 = random.choice(template["items2"])

        # Generate base problem
        base_problem = f"You have one {specific_item1} and one {specific_item2}."

        # Determine number of distractors
        if self.distractor_count_distribution:
            # Use precise distractor count distribution
            cumulative_count = 0
            for distractor_count_str, count in self.distractor_count_distribution.items():
                cumulative_count += count
                if instance_id < cumulative_count:
                    num_distractors = int(distractor_count_str)
                    break
            else:
                # Fallback if instance_id exceeds total
                num_distractors = random.choice(self.available_distractor_counts)
        else:
            # Use available distractor counts or default random
            if self.available_distractor_counts:
                num_distractors = random.choice(self.available_distractor_counts)
            else:
                num_distractors = random.randint(1, self.generation_config.max_distractors)
        
        # Generate distractors
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
            "dataset": "synthetic_misleading_python",
            "category": template["category"],
            "item1": specific_item1,
            "item2": specific_item2,
            "num_distractors": num_distractors,
            "distractors": distractors
        }
        
        # Add target distractor count if it was specified
        if self.distractor_count_distribution:
            metadata["target_distractor_count"] = num_distractors

        return TaskInstance(
            prompt=prompt,
            answer=answer,
            metadata=metadata
        )

    def _get_article(self, word: str) -> str:
        """Determine whether to use 'a' or 'an' based on the word."""
        vowels = 'aeiouAEIOU'
        return 'an' if word[0] in vowels else 'a'

    # Python-specific distractor generation methods

    def _generate_list_comprehension_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with irrelevant list comprehension."""
        num1 = random.randint(3, 10)
        num2 = random.randint(3, 10)

        return (
            f"Your friend shows you a Python code snippet: "
            f"`[x for x in range({num1}, {num2}) if x % 2 == 0]` and says "
            f"this might help calculate the number of {template['category']}."
        )

    def _generate_variable_naming_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with misleading variable names."""
        num1 = random.randint(5, 15)
        num2 = random.randint(5, 15)
        op = random.choice(["+", "-", "*", "//", "%"])

        return (
            f"Your friend writes the following code: "
            f"`{template['category']}_count = {num1} {op} {num2}` but doesn't run it."
        )

    def _generate_function_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with unused function definition."""
        return (
            f"Your friend defines a function: ```python\\n"
            f"def count_{template['category']}(items):\\n"
            f"    return len(items) if isinstance(items, list) else 1\\n```"
        )

    def _generate_complex_calculation_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with complex but irrelevant calculation."""
        return (
            f"Your friend runs: ```python\\n"
            f"import math\\n"
            f"math.factorial(3) // math.gcd(8, 12) + sum(range(4))\\n```"
            f" and says this is related to the {template['category']}."
        )

    def _generate_class_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with class definition."""
        item1 = template['item1'].replace(' ', '_')
        item2 = template['item2'].replace(' ', '_')

        return (
            f"Your friend creates a class: ```python\\n"
            f"class {template['category'].capitalize()}Counter:\\n"
            f"    def __init__(self):\\n"
            f"        self.{item1}_count = 0\\n"
            f"        self.{item2}_count = 0\\n```"
        )

    def _generate_recursive_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with recursive function."""
        return (
            f"Your friend writes a recursive function: ```python\\n"
            f"def count_recursive(n):\\n"
            f"    return 1 if n <= 1 else n + count_recursive(n-1)\\n```"
            f" and claims it counts {template['category']}."
        )

    def _generate_loop_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with loop code."""
        num = random.randint(3, 8)
        return (
            f"Your friend shows you: ```python\\n"
            f"total = 0\\n"
            f"for i in range({num}):\\n"
            f"    total += i\\n"
            f"print(total)\\n```"
            f" and says this is the number of {template['category']}."
        )

    def _generate_exception_handling_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with exception handling."""
        return (
            f"Your friend writes: ```python\\n"
            f"try:\\n"
            f"    count = len({template['category']}_list)\\n"
            f"except NameError:\\n"
            f"    count = 0\\n```"
            f" but {template['category']}_list is undefined."
        )

    def _generate_lambda_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with lambda function."""
        return (
            f"Your friend defines: ```python\\n"
            f"counter = lambda x: len(x) if hasattr(x, '__len__') else 1\\n```"
            f" and says it counts {template['category']}."
        )

    def _generate_decorator_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate distractor with decorator."""
        return (
            f"Your friend uses a decorator: ```python\\n"
            f"@property\\n"
            f"def {template['category']}_count(self):\\n"
            f"    return self._count\\n```"
            f" but never initializes _count."
        )
